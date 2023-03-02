from torch import nn
from transformers import T5ForConditionalGeneration, T5Config
from utils import wipe_between_space, print_rank_0

import os


class T5Gen_Model(nn.Module):
    def __init__(self, model_path, tokenizer, data, dropout):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer  # tokenizer with extended vocabulary
        self.pad_token_id = self.data.pad_token_id
        self.sos_g_token_id = self.data.sos_g_token_id
        self.eos_g_token_id = self.data.eos_g_token_id
        self.max_tgt_len = self.data.max_tgt_len

        print_rank_0('Initializing Huggingface T5 model...')
        t5_config = T5Config.from_pretrained(model_path)
        t5_config.__dict__["dropout"] = dropout
        self.model = T5ForConditionalGeneration.from_pretrained(model_path, config=t5_config, resume_download=True)
        print_rank_0('Resizing Token Embeddings...')
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, src_input, src_mask, tgt_input, tgt_output):
        src_mask = src_mask.type(src_input.type())
        outputs = self.model(input_ids=src_input, attention_mask=src_mask, decoder_input_ids=tgt_input,
                             labels=tgt_output)
        loss = outputs[0]
        return loss

    def batch_prediction(self, src_input, src_mask, return_tensor=False):
        outputs = self.model.generate(input_ids=src_input, attention_mask=src_mask,
                                      decoder_start_token_id=self.sos_g_token_id,
                                      pad_token_id=self.pad_token_id, eos_token_id=self.eos_g_token_id,
                                      max_length=self.max_tgt_len)
        return outputs if return_tensor else self.parse_batch_text(outputs)

    @wipe_between_space
    def parse_batch_text(self, batch_pred_ids):
        res_text_list = []
        for predicted_ids in batch_pred_ids:
            one_pred_ids = []
            for one_id in predicted_ids:
                if one_id in [self.pad_token_id, self.sos_g_token_id, self.eos_g_token_id]:
                    pass
                else:
                    one_pred_ids.append(one_id)
            one_res_text = self.tokenizer.decode(one_pred_ids)
            res_text_list.append(one_res_text)
        return res_text_list

    def save_model(self, ckpt_save_path):
        if not os.path.exists(ckpt_save_path):
            os.mkdir(ckpt_save_path)
        # save model
        self.model.save_pretrained(ckpt_save_path)
        # save tokenizer
        self.tokenizer.save_pretrained(ckpt_save_path)
