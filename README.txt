Recovered ZED SDK modules

All recovery work is stored here in separate module directories.
Start with perception_recovery/INDEX.txt and each module's REPORT.txt.
Depth usage: depth_recovery/PYTORCH_USAGE.txt and RGB_COMPARISON_USAGE.txt.
Depth PTH/ONNX results: depth_recovery/pth_onnx_comparison/summary.json.

Run documented commands from /usr/local/zed. Python package imports now use
backengineer.depth_recovery. Installed SDK libraries, headers and original
resources remain in /usr/local/zed/lib, include and resources.
