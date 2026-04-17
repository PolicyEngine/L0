CI lint job now runs `ruff format --check .` instead of the retired
`black -l 79 --check`. The Makefile was switched to `ruff format` in #40
but the reusable lint workflow was not updated, so every PR's `lint` check
was failing on files already formatted with ruff.
