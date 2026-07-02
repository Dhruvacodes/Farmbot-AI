#!/bin/bash
# Start USB & Depth camera stream server
cd ~/running_frc_models
python3 jetson_camera_server.py --host 0.0.0.0 --port 8090
