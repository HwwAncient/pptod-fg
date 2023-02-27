import os
import torch
import random
import json

from torch.nn.utils import rnn


def map_bool(bool_status):
    if bool_status == 'True':
        return True
    elif bool_status == 'False':
        return False
    else:
        raise Exception('Wrong Bool Status')


# TODO set as params
dataset_name_list = ['advertisegen', 'lcsts_new', 'dulemon_faithfulmatch_v2']


class PRETRAINING_CORPUS:
    def __init__(self, tokenizer, dataset_prefix_path, max_src_len=1024, max_tgt_len=256):
        print('Tokenizer Size is %d' % len(tokenizer))
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.sos_g_token_id = self.tokenizer.convert_tokens_to_ids(['<sos_g>'])[0]
        self.eos_g_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_g>'])[0]

        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        self.train_data_list = self.load_data(dataset_prefix_path, mode='train')
        self.dev_data_list = self.load_data(dataset_prefix_path, mode='dev')
        self.test_data_list = self.load_data(dataset_prefix_path, mode='test')
        self.shuffle_train_data()

        self.train_num = len(self.train_data_list)
        self.dev_num = len(self.dev_data_list)
        self.test_num = len(self.test_data_list)
        print(f'train data number is {self.train_num}')
        print(f'dev data number is {self.dev_num}')
        print(f'test data number is {self.test_num}')

    def load_data(self, dataset_prefix_path, mode="train"):
        all_dataset_list = []
        for data_name in dataset_name_list:
            one_dataset_list = self.parse_one_dataset(dataset_prefix_path, data_name, mode)
            all_dataset_list += one_dataset_list
        return all_dataset_list

    def parse_one_dataset(self, dataset_prefix_path, data_name, mode="train"):
        assert mode in ['train', 'dev', 'test']
        dataset_path = os.path.join(dataset_prefix_path, f"{mode}_{data_name}.json")
        if not os.path.exists(dataset_path):
            print(f'The data from {dataset_path} is not existed!')
            return []

        print('Loading data from {}'.format(dataset_path))
        with open(dataset_path) as f:
            samples = json.load(f)

        all_sample_list = []
        for sample in samples:
            src = sample['src_id_list'][-self.max_src_len:]
            tgt = sample['tgt_id_list']
            if mode == "train":
                tgt = tgt[:-1][:self.max_tgt_len] + [self.eos_g_token_id]
            assert len(src) > 0 and len(tgt) > 0, f"The length of the sample's src or tgt is 0: {sample}!"
            all_sample_list.append((src, tgt))
        return all_sample_list

    def shuffle_train_data(self):
        random.shuffle(self.train_data_list)

    def get_batches(self, batch_size, mode):
        batch_list = []
        if mode == 'train':
            self.shuffle_train_data()
            all_data_list = self.train_data_list
        elif mode == 'dev':
            all_data_list = self.dev_data_list
        elif mode == 'test':
            all_data_list = self.test_data_list
        else:
            raise Exception('Wrong Mode!!!')

        all_input_data_list, all_output_data_list = [], []
        for inp, oup in all_data_list:
            all_input_data_list.append(inp)
            all_output_data_list.append(oup)

        data_num = len(all_input_data_list)
        batch_num = int(data_num / batch_size) + 1

        for i in range(batch_num):
            start_idx, end_idx = i * batch_size, (i + 1) * batch_size
            if start_idx >= data_num:
                break
            end_idx = min(end_idx, data_num)
            one_input_batch_list, one_output_batch_list = [], []
            for idx in range(start_idx, end_idx):
                one_input_batch_list.append(all_input_data_list[idx])
                one_output_batch_list.append(all_output_data_list[idx])
            one_batch = [one_input_batch_list, one_output_batch_list]
            batch_list.append(one_batch)
        out_str = 'Overall Number of datapoints is ' + str(data_num) + \
                  ' Number of ' + mode + ' batches is ' + str(len(batch_list))
        print(out_str)
        return batch_list

    def build_iterator(self, batch_size, mode):
        batch_list = self.get_batches(batch_size, mode)
        for i, batch in enumerate(batch_list):
            yield batch

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
