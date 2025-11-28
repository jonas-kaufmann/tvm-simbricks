#!/bin/bash
set -eux

WORKLOAD_OPTS=(resnet18_v1 resnet34_v1 resnet50_v1)
INFERENCE_DEVICE_CPU_OPTS=(cpu_arm64)
INFERENCE_DEVICE_VTA_OPTS=(vta)
BLOCK_OPTS=(16 32)

# VTA-based inference
for BLOCK_OPT in ${BLOCK_OPTS[@]}; do
  BLOCK_OPT_LOG=$(echo $BLOCK_OPT | awk '{print log($1)/log(2)}')
  cat > /workspaces/tvm-acdsim/3rdparty/vta-hw/config/vta_config.json <<EOF
{
      "TARGET": "simbricks-pci",
      "HW_VER": "0.0.2",
      "LOG_INP_WIDTH": 3,
      "LOG_WGT_WIDTH": 3,
      "LOG_ACC_WIDTH": 5,
      "LOG_BATCH": 0,
      "LOG_BLOCK": ${BLOCK_OPT_LOG},
      "LOG_UOP_BUFF_SIZE": 15,
      "LOG_INP_BUFF_SIZE": 15,
      "LOG_WGT_BUFF_SIZE": 18,
      "LOG_ACC_BUFF_SIZE": 17
}
EOF
  for INFERENCE_DEVICE in ${INFERENCE_DEVICE_VTA_OPTS[@]}; do
    for WORKLOAD_OPT in ${WORKLOAD_OPTS[@]}; do
      python deploy_classification-compile_lib.py ${INFERENCE_DEVICE} cpu_arm64 ${WORKLOAD_OPT} /workspaces/tvm-acdsim/mxnet/
    done
  done
done

# CPU-based inference
for INFERENCE_DEVICE in ${INFERENCE_DEVICE_CPU_OPTS[@]}; do
  for WORKLOAD_OPT in ${WORKLOAD_OPTS[@]}; do
    python deploy_classification-compile_lib.py ${INFERENCE_DEVICE} cpu_arm64 ${WORKLOAD_OPT} /workspaces/tvm-acdsim/mxnet/
  done
done
