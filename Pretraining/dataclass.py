import os
import json
import torch
import argparse

from tqdm import tqdm
from torch.nn.utils import rnn
from utils import print_rank_0


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


# TODO set as params
dataset_name_list = ['advertisegen', 'lcsts_new', 'dulemon_faithfulmatch_v2']


class PRETRAINING_CORPUS:
    def __init__(self, tokenizer, dataset_prefix_path, max_src_len=1024, max_tgt_len=256):
        print_rank_0('Tokenizer Size is %d' % len(tokenizer))
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.sos_g_token_id = self.tokenizer.convert_tokens_to_ids(['<sos_g>'])[0]
        self.eos_g_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_g>'])[0]

        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        self.load_data(dataset_prefix_path, mode='train')
        self.load_data(dataset_prefix_path, mode='dev')
        self.load_data(dataset_prefix_path, mode='test')

    def load_data(self, dataset_prefix_path, mode="train"):
        data_file = os.path.join(dataset_prefix_path, f"{mode}.json")
        if os.path.exists(data_file):
            print_rank_0(f"[{data_file}]: already exists!")
            return

        print_rank_0(f"[{data_file}]: not exists, begin to construct it!")
        with open(data_file, "w", encoding="utf-8") as fw:
            for idx, data_name in enumerate(dataset_name_list):
                self.parse_one_dataset(idx, fw, dataset_prefix_path, data_name, mode)

    def parse_one_dataset(self, idx, fw, dataset_prefix_path, data_name, mode="train"):
        assert mode in ['train', 'dev', 'test']
        dataset_path = os.path.join(dataset_prefix_path, f"{mode}_{data_name}.json")
        if not os.path.exists(dataset_path):
            print_rank_0(f'{idx} -> the dataset from {dataset_path} is not existed!')
            return

        print_rank_0(f'{idx} -> begin to load dataset from {dataset_path}!')
        with open(dataset_path, "r", encoding="utf-8") as fr:  # TODO: save and load each dataset line by line.
            samples = json.load(fr)

        for sample in tqdm(samples):
            src = sample['src_id_list'][-self.max_src_len:]  # TODO truncate beyond instructions
            tgt = sample['tgt_id_list']
            if mode == "train":
                tgt = tgt[:-1][:self.max_tgt_len] + [self.eos_g_token_id]
            assert len(src) > 0 and len(tgt) > 0, f"The length of the sample's src or tgt is 0: {sample}!"
            fw.write(json.dumps({"src": src, "tgt": tgt}) + "\n")

    def pad_batch(self, batch_id_list):
        batch_id_list = [torch.LongTensor(item) for item in batch_id_list]
        batch_tensor = rnn.pad_sequence(batch_id_list, batch_first=True, padding_value=self.pad_token_id)
        batch_mask = torch.ones_like(batch_tensor)
        batch_mask = batch_mask.masked_fill(batch_tensor.eq(self.pad_token_id), 0.0).type(torch.FloatTensor)
        return batch_tensor, batch_mask

    def process_output(self, batch_tgt_id_list):
        batch_tgt_id_list = [torch.LongTensor(item) for item in batch_tgt_id_list]
        batch_tgt_tensor, _ = self.pad_batch(batch_tgt_id_list)
        batch_tgt_input_tensor = batch_tgt_tensor[:, :-1].clone()
        batch_tgt_output_tensor = batch_tgt_tensor[:, 1:].clone()
        return batch_tgt_input_tensor, batch_tgt_output_tensor

    def parse_batch_tensor(self, batch):
        batch_input_id_list, batch_output_id_list = batch
        batch_src_tensor, batch_src_mask = self.pad_batch(batch_input_id_list)
        batch_input, batch_labels = self.process_output(batch_output_id_list)
        batch_labels[batch_labels[:, :] == self.pad_token_id] = -100
        return batch_src_tensor, batch_src_mask, batch_input, batch_labels

    def collate_fn(self, samples):
        src = [sample['src'] for sample in samples]
        tgt = [sample['tgt'] for sample in samples]
        batch = self.parse_batch_tensor(batch=(src, tgt))
        return batch
