#!/bin/bash
# A bash script to apply the formatting for this repo
black -l 120 code/*.py
isort -l 120 code/*.py