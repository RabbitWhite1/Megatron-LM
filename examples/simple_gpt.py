import argparse
import os
from functools import partial
from pathlib import Path

import torch
import torch.distributed
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchgraph.graph.dynamo.tools import dynamo_and_dump

from megatron.core import dist_checkpointing, parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.core.datasets.utils import compile_helpers
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.tokenizer.tokenizer import _NullTokenizer

_SEQUENCE_LENGTH = 64


def initialize_distributed(tensor_model_parallel_size=1, pipeline_model_parallel_size=1):
    parallel_state.destroy_model_parallel()

    # Torch setup for distributed training
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(world_size=world_size, rank=rank)

    # Megatron core distributed training initialization
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size, pipeline_model_parallel_size
    )


def model_provider(tensor_model_parallel_size=1):
    """Build the model."""

    transformer_config = TransformerConfig(
        num_layers=1,
        hidden_size=12,
        num_attention_heads=4,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        sequence_parallel=tensor_model_parallel_size > 1,
        tensor_model_parallel_size=tensor_model_parallel_size,
        deterministic_mode=True,
    )

    gpt_model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=163840,
        max_sequence_length=_SEQUENCE_LENGTH,
    )

    return gpt_model


def get_train_data_iterator():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()
    else:
        compile_helpers()

    config = GPTDatasetConfig(
        random_seed=0,
        sequence_length=_SEQUENCE_LENGTH,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=_NullTokenizer(vocab_size=_SEQUENCE_LENGTH),
    )

    datasets = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [8, None, None], lambda: True, config
    ).build()

    train_dataloader = DataLoader(datasets[0], batch_size=8, shuffle=True)

    train_iterator = iter(train_dataloader)

    return train_iterator


def forward_step_func(data_iterator, model):

    def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):

        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        # If you have data parallel reduce loss across data parallel groups.
        # If pipeline parallel, loss computation is done only in last stage.

        return loss, {'lm loss': loss}

    data = data_iterator.pop(0)
    tokens = data['tokens'].to(device)
    attention_mask = data['attention_mask'].to(device)
    position_ids = data['position_ids'].to(device)
    labels = data['labels'].to(device)
    loss_mask = data['loss_mask'].to(device)

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)

parser = argparse.ArgumentParser("Simple GPT")
parser.add_argument("--tp_size", default=1, type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    initialize_distributed(tensor_model_parallel_size=args.tp_size, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)

    gpt_model = model_provider(tensor_model_parallel_size=args.tp_size)
    device = torch.device("cuda")
    gpt_model.to(device)

    optim = Adam(gpt_model.parameters())

    train_dataset = []
    train_iterator = get_train_data_iterator()
    for _ in range(1):
        train_dataset.append(next(train_iterator))

    forward_backward_func = get_forward_backward_func()

    optim.zero_grad()

    # dynamo = os.environ.get("TG_USING_DYNAMO", "0") == "1"
    dynamo = True

    def fn(model):
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=train_dataset,
            model=model,
            num_microbatches=1,
            seq_length=_SEQUENCE_LENGTH,
            micro_batch_size=1,
            decoder_seq_length=_SEQUENCE_LENGTH,
            forward_only=False,
        )
        return losses_reduced

    if dynamo:
        dirname = os.environ["TG_DUMP_DIRNAME"]
        os.makedirs(dirname, exist_ok=True)
        _, _, _, res = dynamo_and_dump(
            gpt_model,
            fn,
            dirname=dirname,
            formats=['code'],
            rank=torch.distributed.get_rank(),
            compile_model_or_fn="fn",
            return_res=True,
        )
        losses_reduced = res
    else:
        losses_reduced = fn(gpt_model)


    print(f'Losses reduced :  {losses_reduced}')

    torch.distributed.destroy_process_group()
