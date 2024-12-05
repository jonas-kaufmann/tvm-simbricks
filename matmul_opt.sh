#!/bin/bash
export TVM_HOME=/home/xilinx/tvm-simbricks
export VTA_RPC_HOST=127.0.0.1
export VTA_RPC_PORT=9091
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/vta/python

python vta/tutorials/optimize/matrix_multiply_opt.py
