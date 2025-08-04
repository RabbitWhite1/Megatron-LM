#!/bin/bash

# Runs the "345M" parameter model

export CUDA_VISIBLE_DEVICES=6,7

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=`python -c "import os; print(os.environ['CUDA_VISIBLE_DEVICES'].count(',' ) + 1)"`


# BUGS
export FORCE_DISABLE_GRAD_REDUCE_FOR_OUTPUT_LAYER=0
export FORCE_FORGET_SP_LAYERNORM_ALLREDUCE=0
if [ "$FORCE_FORGET_SP_LAYERNORM_ALLREDUCE" -eq 1 ]; then
    SEQUENCE_PARALLEL="--sequence-parallel"
else
    SEQUENCE_PARALLEL=""
fi
export FORCE_FORGET_SP_SWITCHMLP_ALLREDUCE=1
export FORCE_USING_SWITCHMLP=1
if [ "$FORCE_FORGET_SP_SWITCHMLP_ALLREDUCE" -eq 1 ]; then
    # SWITCHMLP_ARGS="--num-experts 1 --sequence-parallel --use-legacy-models --ckpt-format torch"
    SWITCHMLP_ARGS="--num-experts 1 --sequence-parallel "
else
    SWITCHMLP_ARGS=""
fi

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NUM_NODES=1
NODE_RANK=0
GLOBAL_BATCH_SIZE=32
CP_SIZE=1
TP_SIZE=$GPUS_PER_NODE
PP_SIZE=1
DP_SIZE=$(($GPUS_PER_NODE / ($CP_SIZE * $TP_SIZE * $PP_SIZE) * $NUM_NODES))
# GPUS_PER_NODE=$(($DP_SIZE*$TP_SIZE*$PP_SIZE))
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

echo "GPUS_PER_NODE: $GPUS_PER_NODE, NUM_NODES: $NUM_NODES, TP_SIZE: $TP_SIZE, CP_SIZE: $CP_SIZE, PP_SIZE: $PP_SIZE, DP_SIZE: $DP_SIZE, WORLD_SIZE: $WORLD_SIZE"

VOCAB_FILE=../datasets/gpt2/vocab.json
MERGE_FILE=../datasets/gpt2/merges.txt
DATA_PATH=../datasets/gpt2/my-gpt2_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0
export NVTE_UNFUSED_ATTN=0

export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2 

GPT_MODEL_ARGS=(
    --num-layers 1 
    --hidden-size 1536 
    --num-attention-heads 32
    --seq-length 1024 
    --max-position-embeddings 2048 
    --fp16
    # --use-flash-attn
    --attention-backend flash # Can use (flash/fused/unfused/local)
    --transformer-impl local
    # --deterministic-mode
    --qk-layernorm
    --disable-bias-linear
    $SEQUENCE_PARALLEL

    $SWITCHMLP_ARGS
)

TRAINING_ARGS=(
    --micro-batch-size $(($GLOBAL_BATCH_SIZE / $DP_SIZE))
    --global-batch-size $GLOBAL_BATCH_SIZE
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
    --context-parallel-size $CP_SIZE
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
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1  # Required for using transformer engine

# Deterministic settings
export NCCL_ALGO=Ring
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# PyTorch Debug
export NCCL_DEBUG=WARN  # INFO/WARN
# export NCCL_DEBUG_SUBSYS=COLL
# export TORCH_DISTRIBUTED_DEBUG=INFO # INFO/WARN

# Config Transformer Engine
export NVTE_TORCH_COMPILE=0  # Disable any jit.

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
export TG_HACK_FOR_DYNAMO=1

export TG_DUMP_DIRNAME=gpt/dp${DP_SIZE}-tp${TP_SIZE}-cp${CP_SIZE}

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=30
export CUDA_LAUNCH_BLOCKING=1

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    --no-gradient-accumulation-fusion \
    --no-bias-gelu-fusion \
    --no-bias-swiglu-fusion \
    --no-bias-dropout-fusion \
    --disable-tp-comm-overlap-ag \
    --disable-tp-comm-overlap-rs \
    --disable-tp-comm-split-ag \
    --disable-tp-comm-split-rs \
    --disable-tp-comm-bulk-dgrad \
    --disable-tp-comm-bulk-wgrad \
    --no-check-for-nan-in-loss-and-grad \
    --no-align-grad-reduce \
    --distributed-timeout-minutes 1
