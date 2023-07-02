#!/bin/bash
SCRIPT_DIR=$(cd $(dirname $0); pwd)  # Get shell script abs path
IMAGENAME="apop-image"
TAG="latest"
docker build -t ${IMAGENAME}:${TAG} --build-arg USER=$USER --build-arg USER_ID=$UID ${SCRIPT_DIR}
