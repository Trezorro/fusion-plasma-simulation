"""Vendored subset of thuml/Time-Series-Library (giants/Time-Series-Library submodule).

Only what DLinear, PatchTST and iTransformer forecasting need. Kept as close to
upstream as possible; the only edits are:
  - imports rewritten to this package (relative),
  - PatchTST / iTransformer gained a `use_norm` gate (upstream port dropped the flag),
  - SelfAttention_Family trimmed of ReformerLayer / TwoStageAttentionLayer to avoid the
    reformer-pytorch and einops dependencies (unused by these three models).
"""
