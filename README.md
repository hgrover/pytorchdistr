## PyTorch Distributed Training
This repository contains code that goes along with the [Primer on Distributed Training with PyTorch](https://medium.com/@himanshu.grover/quick-primer-on-distributed-training-with-pytorch-ad362d8aa032)

##### 1. Steps and commands for Single node, multi-gpu setup
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=4 --use_env src/main.py

##### 2. Steps and commands for Multi-node, multi-gpu setup
- **Step 1:** Choose a node as master and find an available high port (here, in range 49000-65535) on it for communication with worker nodes (https://unix.stackexchange.com/a/423052):<br />
MASTER_PORT=\`comm -23 <(seq 49000 65535 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | grep '[0-9]\{1,5\}' | sort -u)| shuf | head -n 1\`
- **Step 2:** Set MASTER_ADDR and MASTER_PORT on all nodes for launch utility:<br />
export MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=$MASTER_PORT
- **Step 3:** Launch master node process: <br />
python -m torch.distributed.launch --nnodes=<num nodes> --node_rank=0 --nproc_per_node=<num_gpus_per_node> --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env src/main.py --distributed true
- **Step 4:** Launch worker nodes' processes (run on each node, setting appropriate node_rank): <br />
python -m torch.distributed.launch --nnodes=<num_nodes> --node_rank=<node rank=worker_node_rank> --nproc_per_node=4 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env src/main.py --distributed true
