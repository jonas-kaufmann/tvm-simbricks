# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Deploy Pretrained Vision Model from MxNet on VTA
================================================
**Author**: `Thierry Moreau <https://homes.cs.washington.edu/~moreau/>`_

This tutorial provides an end-to-end demo, on how to run ImageNet classification
inference onto the VTA accelerator design to perform ImageNet classification tasks.
It showcases Relay as a front end compiler that can perform quantization (VTA
only supports int8/32 inference) as well as graph packing (in order to enable
tensorization in the core) to massage the compute graph for the hardware target.
"""

import os
import sys
import time
import random

import numpy as np
import vta
from PIL import Image

import tvm
from tvm.autotvm.measure import measure_methods
from tvm.contrib import graph_executor


def main():
    if len(sys.argv) != 9:
        print(
            "Usage: deploy_detection-infer.py <mxnet_dir> <target_name>"
            " <model_name> <test_image> <batch_size> <repetitions> <debug> <seed>"
        )
        sys.exit(1)

    mxnet_dir = sys.argv[1]
    target_name = sys.argv[2]
    model_name = sys.argv[3]
    test_image = sys.argv[4]
    batch_size = int(sys.argv[5])
    reps = int(sys.argv[6])
    debug = int(sys.argv[7])
    random.seed(int(sys.argv[8]))

    # Load VTA parameters from the 3rdparty/vta-hw/config/vta_config.json file
    env = vta.get_env()

    # prepare required files
    accel_cfg = ""
    if target_name == "vta":
        accel_cfg = f"-{env.BATCH}x{env.BLOCK_OUT}"
    graphlib = f"{mxnet_dir}/graphlib-{model_name}-{target_name}{accel_cfg}.so"

    e2e_start = time.time_ns()

    # Prepare test image for inference
    image = Image.open(test_image).resize((224, 224))
    image = np.array(image) - np.array([123.0, 117.0, 104.0])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    image = np.repeat(image, env.BATCH, axis=0)
    assert (
        batch_size % env.BATCH == 0
    ), f"batch_size={batch_size} env.BATCH={env.BATCH}"

    #######################################################################
    # Connect to tracker or RPC server and request remote inference device.
    # ---------------------------------------------------------------------
    # Tracker handles hands out leases to remote inference devices
    tracker_host = os.environ.get("TVM_TRACKER_HOST", None)
    tracker_port = os.environ.get("TVM_TRACKER_PORT", None)
    # If above are unset, skip Tracker and instead connect to device directly
    device_host = os.environ.get("VTA_RPC_HOST", "127.0.0.1")
    device_port = os.environ.get("VTA_RPC_PORT", "9091")
    assert tvm.runtime.enabled("rpc")

    for i in range(reps):
        sleep_for = random.randint(0, 10)
        print(f"Rep {i} sleeping for {sleep_for} s")
        time.sleep(sleep_for)
        e2e_start = time.time_ns()
        request_start = time.time_ns()
        if tracker_host is None or tracker_port is None:
            remote = tvm.rpc.connect(device_host, int(device_port))
        else:
            remote = measure_methods.request_remote(
                env.TARGET, tracker_host, int(tracker_port), timeout=0
            )
        request_dur = time.time_ns() - request_start

        # Send the inference library over to the remote RPC server
        upload_lib_start = time.time_ns()
        remote.upload(graphlib)
        lib = remote.load_module(os.path.basename(graphlib))
        upload_lib_dur = time.time_ns() - upload_lib_start

        # Request remote device
        ctx = remote.ext_dev(0) if target_name == "vta" else remote.cpu(0)

        ####################################
        # Perform image detection inference.
        # ----------------------------------
        # Graph executor
        m = graph_executor.GraphModule(lib["default"](ctx))

        num_inferences = batch_size // env.BATCH
        inference_start = time.time_ns()
        for j in range(num_inferences):
            # Set the network parameters and inputs
            m.set_input("data", image)
            # Perform inference
            m.run()
            # Get output
            tvm_output = m.get_output(
                0, tvm.nd.empty((env.BATCH, 1000), "float32", remote.cpu(0))
            ).numpy()
        inference_dur = time.time_ns() - inference_start

        # release resources
        remote._sess.get_function("CloseRPCConnection")()

        e2e_dur = time.time_ns() - e2e_start
        print(f"Rep {i}: Requesting remote device {request_dur:_} ns")
        print(f"Rep {i}: Sending and loading model {upload_lib_dur:_} ns")
        print(f"Rep {i}: Pure inference duration {inference_dur:_} ns")
        print(f"Rep {i}: End-to-end latency: {e2e_dur:_} ns")

    if not debug:
        return

    # read classification categories
    synset = eval(open(f"{mxnet_dir}/synset.txt").read())

    # Report top-5 classification results
    for b in range(env.BATCH):
        top_categories = np.argsort(tvm_output[b])
        print(f"\nprediction for sample {b}")
        for i in range(1, 6):
            print(
                f"\t#{i}:{synset[top_categories[-i]]} {tvm_output[b][top_categories[-i]]}"
            )


if __name__ == "__main__":
    main()
