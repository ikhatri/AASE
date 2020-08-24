#!/bin/bash
# A script to run the Darknet YOLOv3 model on a ring front center video and then save the output to an output video
DARKNET=$1 # The first argument is the path to the folder that contains the darknet executable
VIDEO=$2   # The second argument is the path to the folder containing the input video
if [ -z "$3" ]
then
    cd $DARKNET && ./darknet detector demo data/tl.data cfg/yolov3-tl.cfg backup/yolov3-tl_final.weights $VIDEO/rfc.mp4 -out_filename yolo_output.avi && mv yolo_output.avi $VIDEO
else
    cd $DARKNET && ./darknet detector demo data/tl.data cfg/yolov3-tl.cfg backup/yolov3-tl_final.weights $VIDEO/rfc.mp4
fi
