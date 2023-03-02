"""
Copied from https://github.com/PaddlePaddle/PaddleNLP/blob/v2.0.8/examples/text_generation/unimo-text/gen_utils.py#L140
TODO: to use 'post_process_sum'
"""

import torch
import torch.distributed as dist

from contextlib import contextmanager


def post_process_sum(token_ids, tokenizer):
    """Post-process the decoded sequence. Truncate from the first <eos>."""
    eos_pos = len(token_ids)
    for i, tok_id in enumerate(token_ids):
        if tok_id == tokenizer.mask_token_id:
            eos_pos = i
            break
    token_ids = token_ids[:eos_pos]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = tokenizer.merge_subword(tokens)
    special_tokens = ['[UNK]']
    tokens = [token for token in tokens if token not in special_tokens]
    return token_ids, tokens


def wipe_between_space(func):
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        assert isinstance(res, list)
        if not res:
            return res
        assert isinstance(res[0], str)
        return [sent.replace(' ', '') for sent in res]
    return wrapper


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def is_last_rank():
    return dist.get_rank() == (
        dist.get_world_size() - 1)


def print_rank_last(message):
    """If distributed is initialized, print only on last rank."""
    if dist.is_initialized():
        if is_last_rank():
            print(message, flush=True)
    else:
        print(message, flush=True)


def gather_across_procs(tensor, world_size, max_len=None, padding_value=None):
    assert tensor.dim() == 2

    # 1. pad tensor to the same shape
    # TODO dynamic padding strategy
    # max_len = torch.tensor(tensor.size(1), dtype=torch.int64).to(tensor.device)
    # max_len = reduce_max_across_procs(tensor=max_len).item()
    if max_len is not None and padding_value is not None:
        tensor = pad_to_max_len(sequences=tensor, max_len=max_len, padding_value=padding_value)

    # 2. gather tensor across all processes
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    # dist.barrier()  # synchronizes all processes
    dist.all_gather(tensor_list=tensor_list, tensor=tensor)

    # 3. rearrange the tensor due to the DistributedSampler
    tensor = torch.stack(tensor_list, dim=1).view(-1, tensor.size(-1))
    return tensor


def reduce_mean_across_procs(tensor, world_size):
    # dist.barrier()  # synchronizes all processes
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return tensor


def reduce_max_across_procs(tensor):
    # dist.barrier()  # synchronizes all processes
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor


@contextmanager
def torch_distributed_zero_first(rank):
    if rank not in [-1, 0]:
        dist.barrier()  # synchronizes all processes
    yield
    if rank == 0:
        dist.barrier()  # synchronizes all processes


def pad_to_max_len(sequences, max_len, padding_value=0.0):
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    out_dims = (len(sequences), max_len) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        out_tensor[i, :length, ...] = tensor

    return out_tensor
