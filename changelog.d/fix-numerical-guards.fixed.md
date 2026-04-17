Standardize Hard Concrete numerical guards: all three modules now use
`1e-6` as the uniform-sampling epsilon (previously `distributions.py` used
`1e-8`, which underflows at fp16), and clamp `qz_logits`/`log_alpha` to
`[-20, 20]` before sampling, deterministic gates, and penalty computation
so gradients don't vanish on saturating inputs.
