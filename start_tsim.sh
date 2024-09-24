#!/bin/bash

trap 'pkill -P $$' INT

for (( i=0; i<48; i++ ))
do 
    python3 -m vta.exec.rpc_server --key=tsim --tracker=127.0.0.1:9190 &
    sleep 0.1
done

wait
