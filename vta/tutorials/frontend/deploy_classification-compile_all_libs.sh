#!/bin/bash
set -eux

WORKLOAD_OPTS=(resnet18_v1 resnet34_v1 resnet50_v1)
INFERENCE_DEVICE_OPTS=(cpu_arm64 vta)

for WORKLOAD_OPT in ${WORKLOAD_OPTS[@]}; do
  for INFERENCE_DEVICE in ${INFERENCE_DEVICE_OPTS[@]}; do
    python deploy_classification-compile_lib.py ${INFERENCE_DEVICE} cpu_arm64 ${WORKLOAD_OPT} /home/jonask/Repos/tvm-simbricks/mxnet/
  done
done

