#!/bin/bash

set -xe

for model in ssd_mobilenet_v2_cars; do
    python3 build_engine.py ${model}
done
