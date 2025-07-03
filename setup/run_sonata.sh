#!/bin/bash

# Start a virtual X server
Xorg :99 -noreset +extension GLX +extension RANDR +extension RENDER \
  -logfile /var/log/Xorg.99.log \
  -config /etc/X11/10-xorg-headless.conf &

# Start VNC for remote GUI access
x11vnc -display :99 -forever -nopw -shared &

# Allow Docker containers to connect to the X server
xhost +local:docker

# Run the container with ZED and SONATA installed
docker run --runtime=nvidia \
  -e DISPLAY=:99 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --device /dev/video0 \
  --net=host \
  --privileged \
  -it aiemvirt/zed_sonata_combined \
  bash -c "
    source /opt/conda/etc/profile.d/conda.sh && \
    conda activate sonata && \
    cd /root/sonata && \
    echo 'SONATA + ZED ready. To start SONATA training or viewer, run python tools/train.py or tools/viewer.py' && \
    exec bash
  "
