"""
Copied from https://github.com/PaddlePaddle/PaddleNLP/blob/v2.0.8/examples/text_generation/unimo-text/gen_utils.py#L140
TODO: to use 'post_process_sum'
"""


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
