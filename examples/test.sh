# Required for Megatron-LM
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Debug
## NCCL
export NCCL_DEBUG=WARN # INFO/WARN
## Torch
export CRS_LOGGING_LEVEL=WARN       # INFO/WARN
export TORCH_CPP_LOG_LEVEL=WARNING  # INFO/WARNING
export TORCH_DISTRIBUTED_DEBUG=INFO # INFO/WARN
## Dynamo
export TORCHDYNAMO_EXTENDED_DEBUG_CPP=1
export TORCHDYNAMO_VERBOSE=1
export DYNAMO_LOG_LEVEL=WARN
# export TORCH_LOGS="dynamic"  # This is for debugging dynamic shapes..
## Megatron
export MEGATRON_LOGGING_LEVEL=WARN # Related logs: Running collective: ....
export MEGATRON_SHOW_BARRIER_ENTER_EXIT_LOG=0

# Dynamo configs
export DYNAMO_SUPPRESS_ERRORS=0
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1

# Trace configs
export MARIANA_USE_TRACE_TRAINER=1
export TG_LOG_TENSOR=1
export TG_USE_RNG=0
export TG_USE_CUSTOM_OP=1
export TG_USE_COMPILER_DISABLE=0

export TG_USING_DYNAMO=1

# export TG_DUMP_DIRNAME=gpt/paral1
# export nproc_per_node=1
# export TP_SIZE=1

export TG_DUMP_DIRNAME=gpt/paral2
export nproc_per_node=2
export TP_SIZE=2

# export TG_DUMP_DIRNAME=gpt/paral4
# export nproc_per_node=4
# export TP_SIZE=4

# export TG_DUMP_DIRNAME=gpt/paral8
# export nproc_per_node=8
# export TP_SIZE=8

PYTHONPATH=$PYTHON_PATH:./megatron TORCHRUN --nproc-per-node $nproc_per_node examples/simple_gpt.py --tp_size=$TP_SIZE 
