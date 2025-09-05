# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), 
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-09-05 13:20:21

### Added

- Group-wise loss averaging for calibration to balance contributions from targets with different cardinalities
- Improved training output with meaningful error percentages and sparsity statistics

### Changed

- Simplified active weight detection in SparseCalibrationWeights (removed threshold parameter)
- Enhanced verbose output during calibration training to show relative errors and sparsity percentage

## [0.2.0] - 2025-08-31 13:00:17

### Added

- Positive weight constraints for calibration module
- PolicyEngine workflow best practices integration

## [0.1.3] - 2025-08-26 22:20:57

### Added

- PolicyEngine workflow best practices integration

## [0.1.2] - 2025-01-03 00:00:00

### Added

- Sparse calibration stress test example

## [0.1.1] - 2025-01-02 00:00:00

### Fixed

- CI/CD: Separate PR and push workflows

## [0.1.0] - 2025-01-01 00:00:00

### Added

- Initial release of L0 regularization package
- HardConcrete distribution implementation
- L0Linear, L0Conv2d, L0DepthwiseConv2d layers
- SparseMLP for structured sparsity
- L0Gate, SampleGate, FeatureGate, HybridGate for selection tasks
- L0, L2, and combined L0L2 penalty computation
- Temperature scheduling for training stability
- Comprehensive test suite
- GitHub Actions CI/CD pipeline



[0.3.0]: https://github.com/PolicyEngine/L0/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/PolicyEngine/L0/compare/0.1.3...0.2.0
[0.1.3]: https://github.com/PolicyEngine/L0/compare/0.1.2...0.1.3
[0.1.2]: https://github.com/PolicyEngine/L0/compare/0.1.1...0.1.2
[0.1.1]: https://github.com/PolicyEngine/L0/compare/0.1.0...0.1.1
