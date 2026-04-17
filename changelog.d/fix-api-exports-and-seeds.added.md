Export `SparseCalibrationWeights` from the top-level `l0` package (closing
the discoverability gap with `SparseL0Linear`), add an optional `seed`
parameter to both `SparseCalibrationWeights` and `SparseL0Linear` so
`log_alpha` / `log_weight` jitter is reproducible without managing
PyTorch's global RNG, and update `CLAUDE.md` to list `calibration.py` and
`sparse.py` as first-class modules.
