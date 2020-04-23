#!/bin/bash
# A bash script to create a video from the frames of the ring_front_center camera in the argoverse dataset
LOG=$1 # The path to the main log folder
cd $LOG && ffmpeg -f image2 -pattern_type glob -r 30 -i 'ring_front_center/*.jpg' -c:v libx264 -pix_fmt yuv420p rfc.mp4
