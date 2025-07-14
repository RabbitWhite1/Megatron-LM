#!/bin/bash

# Runs the "345M" parameter model

export CUDA_VISIBLE_DEVICES=3,4

export CUDA_DEVICE_MAX_CONNECTIONS=1

TP_SIZE=2
PP_SIZE=1
GPUS_PER_NODE=$(($TP_SIZE*$PP_SIZE))
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# CHECKPOINT_PATH=$1 #<Specify path>
# TENSORBOARD_LOGS_PATH=$2 #<Specify path>
VOCAB_FILE=$1 #<Specify path to file>/gpt2-vocab.json
MERGE_FILE=$2 #<Specify path to file>/gpt2-merges.txt
DATA_PATH=$3 #<Specify path and file prefix>_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 1 
    --hidden-size 512 
    --num-attention-heads 8 
    --seq-length 1024 
    --max-position-embeddings 2048 
    --attention-backend flash # Can use (flash/fused/unfused/local)
    --transformer-impl local
)

TRAINING_ARGS=(
    --micro-batch-size 32 
    --global-batch-size 32 
    # --rampup-batch-size 16 16 5859375 
    --train-iters 1
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    # --fp16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .000
    --lr-decay-iters 430000 
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size $TP_SIZE
	--pipeline-model-parallel-size $PP_SIZE
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    # --save-interval 10000 
    --eval-interval 1000 
    # --save $CHECKPOINT_PATH 
    # --load $CHECKPOINT_PATH 
    --eval-iters 0
    # --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

# Required for Megatron-LM
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_MAX_CONNECTIONS=1
# export MEGATRON_LOGGING_LEVEL=WARN # Related logs: Running collective: ....
export MEGATRON_SHOW_BARRIER_ENTER_EXIT_LOG=0

# Deterministic settings
export NCCL_ALGO=Ring
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# DYNAMO
export TORCHDYNAMO_EXTENDED_DEBUG_CPP=1
export TORCHDYNAMO_VERBOSE=1
export DYNAMO_LOG_LEVEL=WARN

export DYNAMO_SUPPRESS_ERRORS=0
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1

# Trace configs
export MARIANA_USE_TRACE_TRAINER=1
export TG_LOG_TENSOR=1
export TG_USE_RNG=0
export TG_USE_CUSTOM_OP=1
export TG_USE_COMPILER_DISABLE=0
export TG_USING_DYNAMO=1

export TG_DUMP_DIRNAME=gpt/dp1-tp${TP_SIZE}

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    --deterministic-mode \
    --no-gradient-accumulation-fusion \
    --no-bias-gelu-fusion \
    --no-bias-swiglu-fusion \
    --no-bias-dropout-fusion \
    --no-async-tensor-model-parallel-allreduce
