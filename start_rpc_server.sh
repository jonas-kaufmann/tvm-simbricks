#!/bin/bash
export TVM_HOME=/home/xilinx/tvm-simbricks
export VTA_RPC_HOST=127.0.0.1
export VTA_RPC_PORT=9091
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/vta/python

sudo PYTHONPATH=${PYTHONPATH} python3 -m vta.exec.rpc_server --host=${VTA_RPC_HOST} --port=${VTA_RPC_PORT}