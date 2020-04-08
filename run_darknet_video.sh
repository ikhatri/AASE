#!/bin/bash
DARKNET=$1 # The first argument is the path to the folder that contains the darknet executable
VIDEO=$2   # The second argument is the path to the video
cd $DARKNET && ./darknet detector demo data/tl.data cfg/yolov3-tl.cfg backup/yolov3-tl_final.weights $VIDEO
