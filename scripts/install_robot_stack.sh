#!/usr/bin/env bash
set -euo pipefail

echo "== Installing optional robot stack layer =="
python -m pip install -r requirements/robot.txt

echo
echo "Robot stack may require admin privileges or lab-specific SDK setup."
echo "TODO:"
echo "- Install ROS 2 from the official Ubuntu packages or lab image."
echo "- Install the ZED SDK from Stereolabs."
echo "- Install any required udev rules, USB permissions, and vendor drivers."
echo "- Verify Lite6/xArm connectivity on the target lab machine."
