Fix `HardConcrete._deterministic_gates` to apply the temperature scaling so
that `eval()` output matches `_sample_gates`, `get_penalty`, and
`get_active_prob`. Previously the deterministic branch used
`sigmoid(qz_logits)` without dividing by `temperature`, producing a 4x
distortion for PolicyEngine's default `temperature=0.25` and silently
ignoring `TemperatureScheduler` updates at eval time.
