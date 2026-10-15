# Security

## What this project is

Hemolux is an experimental screening aid, not a medical device. Its model runs
entirely in the browser through `onnxruntime-web` (WebAssembly): a photograph is
read by your own device and is not uploaded to run an estimate. It is not
certified for clinical use and must not be used to make a decision about
anyone's health. What has actually been measured, including what failed, is in
[EVALS.md](EVALS.md).

## Reporting a vulnerability

Report it privately, through
[GitHub private vulnerability reporting](https://github.com/Bhagyansh07/hemolux/security/advisories/new),
rather than in a public issue. Include:

- what you did, and what you expected to happen;
- the commit or URL where it reproduces;
- whether you have a working exploit or are describing a concern.

If private reporting is not available to you, open an issue that describes the
problem without the detail an attacker would need, and ask for a contact.

Please give a reasonable window before disclosing publicly. This is a
single-author project with no response SLA; if you have not heard back, say so
on the same thread rather than assuming the report was seen.

## Scope

**In scope**

- Script injection, XSS, or supply-chain tampering in `app/web/`.
- Anything in the two Workers under `functions/` or in the D1 schema that lets
  one person's screening record be read or modified by someone else. A record
  carries an inferred haemoglobin value and a capture timestamp; that is
  health-adjacent data even though it carries no name.
- A build or deploy path that would let an attacker alter the published static
  site or the ONNX graph it ships.
- A committed secret, credential, or personal data in the repository itself.

**Out of scope**

- The Eyes Defy Anemia corpus and its distribution. It comes from Kaggle, not
  from this repository, and `data/raw/` is ignored for exactly that reason.
- The model being wrong. An inaccurate estimate is a correctness problem, not a
  security vulnerability, and it belongs in [EVALS.md](EVALS.md).
- The disclaimer being ignored. The software states plainly that it is not a
  diagnosis.

## Handling of reported issues

Reports are triaged in the order they arrive, publicly tracked only after a fix
is available, and credited in the release notes unless you ask otherwise. A
report that cannot be reproduced will say so, with what was checked.

## Dependencies

`pyproject.toml` pins exact versions because this project reports numbers, and a
silent version change is a silent change to the numbers. `npm` dependencies for
the web front end are locked. Run `pip list --outdated` and `npm audit` before a
release rather than trusting that a lockfile is proof.

## Safe use

- Do not put patient data in this repository, its issue tracker, or its
  screenshots. `data/raw/`, `*.zip`, `kaggle.json`, and `predictions.csv` are
  already ignored because they would otherwise leak identifiable images and lab
  values.
- Treat anything committed under `artifacts/reports/` as aggregate-only. The
  one file that would have carried a per-patient row is ignored on purpose; see
  the comment above that rule in `.gitignore`.
