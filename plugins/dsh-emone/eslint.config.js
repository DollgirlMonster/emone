import js from "@eslint/js";
import globals from "globals";

/**
 * ESLint flat config for this package.
 *
 * The plugins read operator configuration, on-disk harness files and model
 * output, so the recommended set is the floor rather than the whole standard:
 * the extra rules below are the ones whose failure mode has shown up in this
 * codebase (a comparison that `==` coerced, a binding that was never used
 * because the wrong one was read, a non-Error throw that loses the stack).
 */
export default [
  { ignores: ["node_modules/**"] },
  js.configs.recommended,
  {
    files: ["**/*.js"],
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      globals: { ...globals.node },
    },
    rules: {
      eqeqeq: ["error", "smart"],
      "no-var": "error",
      "prefer-const": "error",
      "no-throw-literal": "error",
      "no-return-await": "error",
      "require-atomic-updates": "error",
      // `PeerTable.list()` drops the raw gossip payload with
      // `.map(({ gossip, ...rest }) => ...)`: the name exists to omit the
      // field, and the rest-sibling exemption is the documented way to say so.
      // Every other unused binding is still an error.
      "no-unused-vars": ["error", { ignoreRestSiblings: true }],
    },
  },
];
