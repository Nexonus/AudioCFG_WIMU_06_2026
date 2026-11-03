"""E1/E2 metrics (PROJECT_DESIGN.md §5.1, §5.2, §6).

Pure-numpy cores (diversity, CLAP calibration, FD, KL, copy detection) with injectable
embedders. Heavy model packages are imported lazily inside the embedders, so importing this
package (as ``audiocfg.cli`` does at startup) stays cheap.
"""
