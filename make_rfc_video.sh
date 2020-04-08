#!/bin/bash
ffmpeg -f image2 -pattern_type glob -r 30 -i 'ring_front_center/*.jpg' -c:v libx264 -pix_fmt yuv420p rfc.mp4
