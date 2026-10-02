"""Loss-function tests.

Every expected value is a closed form from the definition, because these losses
are easy to write tests for that only prove ``loss(a) == loss(a)``.

The class this file spends most of its time on is
:class:`~hemolux.losses.InverseFrequencyWeights`, because three separate things
about it were wrong at some point and only one of them was visible in a training
log:

* **The weights were not normalised.** The raw ``1/f_c`` averages, weighted by
  the class frequencies, to exactly the number of classes::

      sum_c  f_c * (1 / f_c) = K

  So a 2-class head trained at learning rate 1e-3 was really taking 2e-3-sized
  steps, the 12-bin ordinal head 12e-3, and the C1 comparison between heads was
  measuring the loss scale rather than the head. That is invisible in a loss
  curve -- it just looks like a fast learner. The fix normalises to unit mean, and
  ``TestNormalisedWeights`` checks the identity that pins it.

* **Absent classes broke the call.** Inferring the class count as ``max + 1``
  yields three weights for a four-class severity head whenever the training split
  happens to contain no severe case, and ``cross_entropy`` rejects a weight
  vector that does not cover every class::

      RuntimeError: weight tensor should be defined either for all 4 classes or
      no classes but got weight tensor of shape: [3]

  This is not hypothetical: the whole corpus holds so few severe cases that a
  given 131-patient split can contain none.

* **Float targets raised instead of indexing.** ``tensor[float_tensor]`` is an
  error, so a float binary label of 0.0 would have failed with an opaque
  ``IndexError`` from deep inside tensor indexing.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.nn.functional import cross_entropy, one_hot, smooth_l1_loss

from hemolux.losses import (
    FocalLoss,
    HuberRegressionLoss,
    InverseFrequencyWeights,
    SoftTargetCrossEntropy,
    WeightedCrossEntropy,
    combined_multitask_loss,
)

# --------------------------------------------------------------------------- #
# Huber
# --------------------------------------------------------------------------- #


class TestHuberRegressionLoss:
    def test_is_quadratic_inside_delta(self) -> None:
        """For |e| <= delta, smooth L1 is 0.5 * e^2 / delta. At delta=1 that is
        0.5 * e^2."""
        loss = HuberRegressionLoss(delta=1.0)
        e = 0.4
        got = loss(torch.tensor([0.0 + e]), torch.tensor([0.0]))
        assert got.item() == pytest.approx(0.5 * e**2, rel=1e-6)

    def test_is_linear_beyond_delta(self) -> None:
        """Past delta the value is ``|e| - 0.5 * delta``, so a 5 g/dL error costs
        4.5 rather than the 12.5 the quadratic branch would have charged. One bad
        frame cannot dominate a batch of accurate ones."""
        loss = HuberRegressionLoss(delta=1.0)
        assert loss(torch.tensor([5.0]), torch.tensor([0.0])).item() == pytest.approx(4.5)

    def test_beyond_delta_the_gradient_stops_growing(self) -> None:
        """This is the whole reason for using it here. Below 1 g/dL -- the
        clinical acceptability band for point-of-care devices -- the penalty
        grows quadratically; beyond it, linearly. One 5 g/dL error cannot dominate
        a batch of accurate predictions."""
        loss = HuberRegressionLoss(delta=1.0)
        assert loss(torch.tensor([0.5]), torch.tensor([0.0])).item() == pytest.approx(0.125)
        assert loss(torch.tensor([1.0]), torch.tensor([0.0])).item() == pytest.approx(0.5)
        # Past delta the value is |e| - 0.5, so a 5-unit error costs 4.5, not 12.5.
        assert loss(torch.tensor([5.0]), torch.tensor([0.0])).item() == pytest.approx(4.5)

    def test_is_symmetric_in_the_sign_of_the_error(self) -> None:
        loss = HuberRegressionLoss()
        under = loss(torch.tensor([1.5]), torch.tensor([0.0])).item()
        over = loss(torch.tensor([0.0]), torch.tensor([1.5])).item()
        assert under == pytest.approx(over)

    def test_zero_error_costs_nothing(self) -> None:
        assert HuberRegressionLoss()(torch.tensor([12.3]), torch.tensor([12.3])).item() == pytest.approx(0.0)

    def test_reported_in_grams_per_decilitre(self) -> None:
        """A 1 g/dL error costs 0.5 with delta=1, which is only interpretable
        because the unit is g/dL and not a normalised quantity."""
        loss = HuberRegressionLoss(delta=1.0)
        assert loss(torch.tensor([14.0]), torch.tensor([13.0])).item() == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Soft-target cross entropy
# --------------------------------------------------------------------------- #


class TestSoftTargetCrossEntropy:
    def test_one_hot_target_reduces_to_ordinary_cross_entropy(self) -> None:
        logits = torch.randn(6, 12)
        targets = torch.randint(0, 12, (6,))
        soft = one_hot(targets, num_classes=12).float()
        assert SoftTargetCrossEntropy()(logits, soft).item() == pytest.approx(
            cross_entropy(logits, targets).item(), rel=1e-6
        )

    def test_is_minimised_when_the_prediction_equals_the_soft_target(self) -> None:
        """The property that keeps an ordinal head calibrated. With a one-hot
        target the loss is indifferent to how the remaining mass is placed; with a
        soft target, matching it exactly is optimal."""
        soft = torch.tensor([[0.1, 0.7, 0.2]])
        logits = (soft * 6.0).log()
        at_target = SoftTargetCrossEntropy()(logits, soft).item()
        shifted = SoftTargetCrossEntropy()((soft * 6.0 + 1.0).log(), soft).item()
        assert at_target < shifted

    def test_matches_the_entropy_when_prediction_equals_target(self) -> None:
        """Cross entropy against the target's own distribution is its entropy."""
        soft = torch.tensor([[0.2, 0.5, 0.3]])
        logits = (soft * 6.0).log()
        expected = -float((soft * soft.log()).sum())
        assert SoftTargetCrossEntropy()(logits, soft).item() == pytest.approx(expected, abs=1e-6)

    def test_is_non_negative(self) -> None:
        assert SoftTargetCrossEntropy()(torch.randn(20, 5) * 10, torch.softmax(torch.randn(20, 5), -1)).item() >= 0.0

    def test_reduces_over_the_batch_not_the_classes(self) -> None:
        """One scalar out, so it composes with the other heads' terms."""
        assert SoftTargetCrossEntropy()(torch.randn(7, 4), torch.softmax(torch.randn(7, 4), -1)).ndim == 0

    def test_grows_when_the_target_mass_moves_to_the_wrong_bin(self) -> None:
        soft = torch.tensor([[0.85, 0.15]])
        good = (soft * 6.0).log()
        bad = torch.tensor([[0.2, 5.5]])
        assert SoftTargetCrossEntropy()(good, soft).item() < SoftTargetCrossEntropy()(bad, soft).item()


# --------------------------------------------------------------------------- #
# Inverse-frequency weights
# --------------------------------------------------------------------------- #


class TestInverseFrequencyWeights:
    def test_balanced_classes_all_get_weight_one(self) -> None:
        w = InverseFrequencyWeights(torch.tensor([0, 1, 0, 1]), n_classes=2)
        assert torch.allclose(w.weights, torch.ones(2), atol=1e-5)

    def test_rare_class_outweighs_the_common_one(self) -> None:
        """9 normal, 1 anaemic."""
        targets = torch.tensor([0] * 9 + [1])
        w = InverseFrequencyWeights(targets, n_classes=2)
        assert w.weights[1] > w.weights[0]

    def test_weights_are_normalised_to_unit_mean(self) -> None:
        """The defect this guards.

        Weighted by the class frequencies, the raw ``1/f_c`` averages to exactly
        K -- 2 for a binary head, 12 for the ordinal one. Normalising makes the
        weighted mean 1.0, so weighting changes which samples are emphasised
        without silently rescaling the effective learning rate.
        """
        targets = torch.tensor([0] * 90 + [1] * 8 + [2] * 2)
        w = InverseFrequencyWeights(targets, n_classes=3)
        freq = torch.bincount(targets.long(), minlength=3).float()
        freq = freq / freq.sum()
        assert (freq * w.weights).sum().item() == pytest.approx(1.0, abs=1e-5)

    @pytest.mark.parametrize(
        "counts",
        [(90, 8, 2), (1, 1), (50, 30, 15, 5), (100,), (*([1] * 9), 50), (999, 1)],
    )
    def test_normalisation_holds_for_any_distribution(self, counts: tuple[int, ...]) -> None:
        """Not just one distribution. Any imbalance would have rescaled the step
        size differently, which is what confounded the head comparison."""
        targets = torch.cat([torch.full((c,), i) for i, c in enumerate(counts)])
        w = InverseFrequencyWeights(targets, n_classes=len(counts))
        freq = torch.bincount(targets.long(), minlength=len(counts)).float()
        freq = freq / freq.sum()
        assert (freq * w.weights).sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_raw_inverse_frequency_really_would_have_averaged_to_k(self) -> None:
        """The identity behind the normalisation, asserted directly so the reason
        for the division is checkable rather than folklore."""
        counts = torch.tensor([90.0, 8.0, 2.0])
        freq = counts / counts.sum()
        assert (freq * (1.0 / freq)).sum().item() == pytest.approx(3.0, abs=1e-4)

    def test_ratio_between_classes_is_preserved_by_normalising(self) -> None:
        """Normalising is a single scalar division, so the ordering and the
        relative gap between classes are untouched."""
        targets = torch.tensor([0] * 90 + [1] * 10)
        w = InverseFrequencyWeights(targets, n_classes=2)
        # 90/10 frequency ratio, so the rare class's raw weight is 9x the common one.
        assert (w.weights[1] / w.weights[0]).item() == pytest.approx(9.0, rel=1e-4)

    def test_absent_class_does_not_break_cross_entropy(self) -> None:
        """Four bands, no severe case in this split.

        Inferring the count as ``max + 1`` would produce three weights and
        ``cross_entropy`` would raise::

            RuntimeError: weight tensor should be defined either for all 4
            classes or no classes but got weight tensor of shape: [3]
        """
        targets = torch.tensor([0] * 30 + [1] * 6 + [2] * 2)
        w = InverseFrequencyWeights(targets, n_classes=4)
        assert w.weights.shape == (4,)
        logits = torch.randn(38, 4)
        loss = cross_entropy(logits, targets.long(), weight=w.weights)
        assert torch.isfinite(loss)

    def test_absent_class_gets_the_smallest_weight_not_the_largest(self) -> None:
        """Raw 1.0 before normalisation, which after the unit-mean division
        becomes the smallest weight of the four.

        The point is what it is *not*: ``1/f`` for an absent class is unbounded,
        and a large weight for a class with no members would be a large number
        multiplied by zero. With counts (30, 6, 2, 0) the weighted mean is exactly
        3, so the absent class lands at 1/3 -- below every present class.
        """
        targets = torch.tensor([0] * 30 + [1] * 6 + [2] * 2)
        w = InverseFrequencyWeights(targets, n_classes=4)
        assert w.weights[3].item() == pytest.approx(1.0 / 3.0, rel=1e-5)
        assert w.weights[3] < w.weights[:3].min()
        assert torch.isfinite(w.weights).all()

    def test_class_count_is_inferred_when_not_given(self) -> None:
        w = InverseFrequencyWeights(torch.tensor([0, 1, 2]))
        assert w.weights.shape == (3,)

    def test_inference_misses_a_trailing_absent_class_and_n_classes_does_not(self) -> None:
        """Both shapes are reachable; the second is the one that used to crash."""
        inferred = InverseFrequencyWeights(torch.tensor([0, 1, 2]))
        explicit = InverseFrequencyWeights(torch.tensor([0, 1, 2]), n_classes=4)
        assert inferred.weights.shape == (3,)
        assert explicit.weights.shape == (4,)

    def test_empty_targets_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty tensor"):
            InverseFrequencyWeights(torch.tensor([], dtype=torch.long))

    def test_out_of_range_index_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            InverseFrequencyWeights(torch.tensor([0, 1, 5]), n_classes=3)

    def test_non_positive_class_count_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            InverseFrequencyWeights(torch.tensor([0]), n_classes=0)

    def test_all_weights_are_finite_and_positive(self) -> None:
        """A single class with one member would give ``1/(1 + eps)``; the epsilon
        must not produce an infinity."""
        w = InverseFrequencyWeights(torch.tensor([0]), n_classes=1)
        assert torch.isfinite(w.weights).all()
        assert (w.weights > 0).all()

    def test_call_maps_targets_to_per_sample_weights(self) -> None:
        targets = torch.tensor([0] * 9 + [1])
        w = InverseFrequencyWeights(targets, n_classes=2)
        per_sample = w(targets)
        assert per_sample.shape == targets.shape
        assert per_sample[0].item() == pytest.approx(w.weights[0].item())
        assert per_sample[-1].item() == pytest.approx(w.weights[1].item())

    def test_call_accepts_float_binary_labels(self) -> None:
        """``tensor[float_tensor]`` raises, so the cast has to happen here. Left to
        the callers, one that forgets produces an opaque IndexError from deep
        inside tensor indexing."""
        w = InverseFrequencyWeights(torch.tensor([0, 1, 0, 1]), n_classes=2)
        per_sample = w(torch.tensor([0.0, 1.0]))
        assert per_sample.tolist() == pytest.approx(
            [w.weights[0].item(), w.weights[1].item()]
        )

    def test_to_returns_the_tensor_not_self(self) -> None:
        """The torch idiom is ``module.to(device)`` returning the module. Returning
        ``self`` here would read like that idiom and quietly do nothing useful."""
        w = InverseFrequencyWeights(torch.tensor([0, 1]), n_classes=2)
        moved = w.to("cpu")
        assert isinstance(moved, torch.Tensor)

    def test_to_the_same_device_is_a_no_op_in_value(self) -> None:
        w = InverseFrequencyWeights(torch.tensor([0, 1, 0]), n_classes=2)
        assert torch.allclose(w.to("cpu"), w.weights)


# --------------------------------------------------------------------------- #
# Weighted cross entropy
# --------------------------------------------------------------------------- #


class TestWeightedCrossEntropy:
    def _case(self, n: int = 8, k: int = 3, seed: int = 0):
        torch.manual_seed(seed)
        return torch.randn(n, k), torch.randint(0, k, (n,))

    def test_without_weights_it_is_plain_cross_entropy(self) -> None:
        logits, targets = self._case()
        assert WeightedCrossEntropy()(logits, targets).item() == pytest.approx(
            cross_entropy(logits, targets).item(), rel=1e-6
        )

    def test_upweighting_the_wrong_predictions_raises_the_loss(self) -> None:
        """The sanity check that the weights reach the reduction at all."""
        logits, targets = self._case()
        unweighted = WeightedCrossEntropy()(logits, targets).item()
        biased = WeightedCrossEntropy(weights=torch.tensor([5.0, 0.2, 0.2]))(logits, targets).item()
        assert biased > unweighted

    def test_equal_weights_reproduce_the_unweighted_loss(self) -> None:
        logits, targets = self._case()
        ones = WeightedCrossEntropy(weights=torch.ones(3))(logits, targets).item()
        assert ones == pytest.approx(cross_entropy(logits, targets).item(), rel=1e-6)

    def test_a_short_weight_vector_fails_loudly(self) -> None:
        """This wrapper indexes the weight vector itself rather than delegating to
        ``F.cross_entropy``, so it raises ``IndexError``, not ``RuntimeError``.

        ``TestShortWeightVectorIsWhatCrossEntropyRejects`` below shows the runtime
        error quoted in the ``n_classes`` docstring, which comes from torch
        directly. Both have to fail -- silently broadcasting three weights across
        a four-class problem is the outcome neither permits.
        """
        logits, targets = self._case(n=8, k=4)
        short = WeightedCrossEntropy(weights=torch.ones(3))
        with pytest.raises(IndexError):
            short(logits, targets)

    def test_short_weight_vector_is_what_cross_entropy_rejects(self) -> None:
        """The exact error quoted in ``InverseFrequencyWeights.__init__``, which is
        what happens when the class count is inferred as ``max + 1`` and the
        highest band happens to be absent from the training split.

        Asserted against torch directly so the docstring's quote stays true: if a
        future torch version changes the wording, this fails and the docstring is
        corrected rather than left describing an error nobody can reproduce.
        """
        targets = torch.tensor([0] * 30 + [1] * 6 + [2] * 2)
        inferred = InverseFrequencyWeights(targets)  # max + 1 -> three classes
        assert inferred.weights.shape == (3,)

        logits = torch.randn(38, 4)
        with pytest.raises(RuntimeError, match="weight tensor"):
            cross_entropy(logits, targets.long(), weight=inferred.weights)

        # And with n_classes supplied the same call succeeds.
        explicit = InverseFrequencyWeights(targets, n_classes=4)
        assert torch.isfinite(cross_entropy(logits, targets.long(), weight=explicit.weights))

    def test_weights_are_registered_as_a_buffer_so_they_follow_the_module(self) -> None:
        module = WeightedCrossEntropy(weights=torch.ones(3))
        assert "weights" in dict(module.named_buffers())


# --------------------------------------------------------------------------- #
# Focal loss
# --------------------------------------------------------------------------- #


class TestFocalLoss:
    def test_gamma_zero_is_ordinary_cross_entropy(self) -> None:
        torch.manual_seed(1)
        logits, targets = torch.randn(10, 4), torch.randint(0, 4, (10,))
        assert FocalLoss(gamma=0.0)(logits, targets).item() == pytest.approx(
            cross_entropy(logits, targets).item(), rel=1e-5
        )

    def test_down_weights_easy_examples_relative_to_hard_ones(self) -> None:
        """``(1 - p)^gamma`` is ~1 for a hard example and ~0 for an easy one, so
        the loss concentrates on the hard tail. That is the whole point: the
        alternative to training this is not reporting sensitivity at all, which is
        not acceptable in a screening tool."""
        torch.manual_seed(2)
        logits, targets = torch.randn(64, 5), torch.randint(0, 5, (64,))
        easy = FocalLoss(gamma=0.0)(logits, targets).item()
        focal = FocalLoss(gamma=2.0)(logits, targets).item()
        assert focal < easy

    def test_confident_and_correct_costs_almost_nothing(self) -> None:
        logits = torch.tensor([[12.0, -12.0]])
        targets = torch.tensor([0])
        assert FocalLoss(gamma=2.0)(logits, targets).item() < 1e-3

    def test_confident_and_wrong_is_heavily_penalised(self) -> None:
        logits = torch.tensor([[12.0, -12.0]])
        targets = torch.tensor([1])
        assert FocalLoss(gamma=2.0)(logits, targets).item() > 20.0

    def test_weights_apply_when_supplied(self) -> None:
        torch.manual_seed(3)
        logits, targets = torch.randn(8, 4), torch.randint(0, 4, (8,))
        plain = FocalLoss(gamma=2.0)(logits, targets).item()
        weighted = FocalLoss(gamma=2.0, weights=torch.tensor([4.0, 0.25, 0.25, 0.25]))(logits, targets).item()
        assert weighted != pytest.approx(plain, rel=1e-9)

    def test_weights_must_cover_every_class(self) -> None:
        logits = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,))
        with pytest.raises(IndexError):
            FocalLoss(gamma=2.0, weights=torch.ones(3))(logits, targets)

    def test_output_is_a_scalar(self) -> None:
        assert FocalLoss()(torch.randn(9, 3), torch.randint(0, 3, (9,))).ndim == 0


# --------------------------------------------------------------------------- #
# Multitask
# --------------------------------------------------------------------------- #


class TestCombinedMultitaskLoss:
    def _inputs(self, n: int = 12, k: int = 6):
        torch.manual_seed(7)
        return torch.randn(n, k), torch.softmax(torch.randn(n, k), -1), torch.randn(n), torch.randn(n) * 2 + 12

    def test_is_the_ordinal_term_plus_a_weighted_huber_term(self) -> None:
        logits, y_soft, scalar, hb = self._inputs()
        got = combined_multitask_loss(logits, y_soft, scalar, hb, scalar_weight=0.5)
        expected = SoftTargetCrossEntropy()(logits, y_soft) + 0.5 * smooth_l1_loss(scalar, hb, beta=1.0)
        assert got.item() == pytest.approx(expected.item(), rel=1e-6)

    def test_zero_scalar_weight_leaves_only_the_ordinal_term(self) -> None:
        logits, y_soft, scalar, hb = self._inputs()
        got = combined_multitask_loss(logits, y_soft, scalar, hb, scalar_weight=0.0)
        assert got.item() == pytest.approx(SoftTargetCrossEntropy()(logits, y_soft).item(), rel=1e-6)

    def test_doubling_the_scalar_weight_doubles_its_contribution(self) -> None:
        logits, y_soft, scalar, hb = self._inputs()
        ordinal = SoftTargetCrossEntropy()(logits, y_soft).item()
        one = combined_multitask_loss(logits, y_soft, scalar, hb, scalar_weight=1.0).item()
        two = combined_multitask_loss(logits, y_soft, scalar, hb, scalar_weight=2.0).item()
        assert (two - ordinal) == pytest.approx(2 * (one - ordinal), rel=1e-5)

    def test_huber_delta_flows_through(self) -> None:
        logits, y_soft, scalar, hb = self._inputs()
        narrow = combined_multitask_loss(logits, y_soft, scalar, hb, huber_delta=0.5).item()
        wide = combined_multitask_loss(logits, y_soft, scalar, hb, huber_delta=5.0).item()
        assert narrow != pytest.approx(wide, rel=1e-9)

    def test_is_a_scalar(self) -> None:
        logits, y_soft, scalar, hb = self._inputs()
        assert combined_multitask_loss(logits, y_soft, scalar, hb).ndim == 0

    def test_is_finite_for_a_confident_wrong_prediction(self) -> None:
        """Focal loss blows up here; the ordinal arm must not."""
        logits = torch.tensor([[20.0, -20.0, -20.0, -20.0]])
        y_soft = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        scalar = torch.tensor([20.0])
        hb = torch.tensor([5.0])
        assert math.isfinite(combined_multitask_loss(logits, y_soft, scalar, hb).item())
