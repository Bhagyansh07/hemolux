/* Dumps both dictionaries so tests/test_i18n_js.py can check parity and that
 * every key the shell references actually exists. No DOM is needed; importing
 * the module is enough. */
import { MESSAGES } from "../app/web/src/i18n.js";

process.stdout.write(
  JSON.stringify({
    en: Object.keys(MESSAGES.en).sort(),
    hi: Object.keys(MESSAGES.hi).sort(),
  }),
);
