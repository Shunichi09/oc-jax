#!/usr/bin/env bash
LOCAL_WORKDIR=$(dirname $(pwd))
CONTAINER_PATH="/home/${USER}/apop"
docker run --gpus all -d \
    --name apop-container \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM="1" \
    -e QT_LOGGING_RULES='*.debug=false;qt.qpa.*=false' \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v "/dev:/dev" \
    --workdir=${CONTAINER_PATH} \
    --rm \
    -v ${LOCAL_WORKDIR}:${CONTAINER_PATH} \
    apop-image \
    tail -f /dev/null
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' apop-container`
docker exec -it apop-container bash
