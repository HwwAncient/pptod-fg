import torch
import argparse
import torch.nn as nn
import random
import math
import numpy as np

from tqdm import tqdm
from dataclass import PRETRAINING_CORPUS
from evaluation import evaluate_by_loss, evaluate_by_bleu
from modelling.T5Model import T5Gen_Model
from transformers import BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup


def set_seed(seed):
    """ fix random seed """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def parse_config():
    parser = argparse.ArgumentParser()

    # data specification
    parser.add_argument('--dataset_prefix_path', type=str, help='the path where all datasets are stored.')
    parser.add_argument("--max_src_len", type=int, default=1024, help="The maximum length of input sequence.")
    parser.add_argument("--max_tgt_len", type=int, default=256, help="The maximum length of output sequence.")

    # model configuration
    parser.add_argument('--model_name', type=str, help='t5-small or t5-base or t5-large')

    # training configuration
    parser.add_argument("--seed", type=int, default=11, help="The number of seed to fix random operations.")
    parser.add_argument("--max_save_num", type=int, default=10, help="The max number of saved checkpoints.")
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size_per_gpu", type=int, default=4, help='Batch size for each gpu.')
    parser.add_argument("--number_of_gpu", type=int, default=8, help="Number of available GPUs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="gradient accumulation step.")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_path", type=str, help="directory to save the model parameters.")
    parser.add_argument("--save_ckpt_name", type=str, help="the name under which to save the pre-trained model. small or base or large")
    return parser.parse_args()


if __name__ == '__main__':
    if torch.cuda.is_available():
        print('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    multi_gpu_training = False
    if cuda_available:
        if torch.cuda.device_count() > 1:
            multi_gpu_training = True
            print('Using Multi-GPU training, number of GPU is {}'.format(torch.cuda.device_count()))
        else:
            print('Using single GPU training.')
    else:
        pass

    args = parse_config()
    device = torch.device('cuda')

    # set seed
    set_seed(seed=args.seed)

    print('Start loading data...')
    preprocessed_tokenizer_path = args.dataset_prefix_path + r'/tokenizer_with_special_token/'
    tokenizer = BertTokenizer.from_pretrained(preprocessed_tokenizer_path)

    data = PRETRAINING_CORPUS(tokenizer=tokenizer, dataset_prefix_path=args.dataset_prefix_path,
                              max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len)
    print('Data Loaded.')

    print('Start loading model...')
    model = T5Gen_Model(model_path=args.model_name, tokenizer=tokenizer, data=data, dropout=args.dropout)

    if cuda_available:
        if multi_gpu_training:
            model = nn.DataParallel(model)  # multi-gpu training
        else:
            pass
        model = model.to(device)
    else:
        pass
    print('Model loaded')

    # organize optimizer
    train_batch_num_per_epoch = math.ceil(data.train_num / (args.number_of_gpu * args.batch_size_per_gpu))
    actual_total_steps = math.ceil(train_batch_num_per_epoch / args.gradient_accumulation_steps) * args.num_train_epochs
    print(f'train_batch_num_per_epoch: {train_batch_num_per_epoch}, '
          f'actual_total_steps: {actual_total_steps}, num_train_epochs: {args.num_train_epochs}')
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    print('Use AdamW Optimizer for Training.')
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=actual_total_steps)
    optimizer.zero_grad()

    global_step, best_dev_loss, best_dev_bleu = 0, 1e10, 0
    for epoch in range(1, args.num_train_epochs + 1):
        model.train()
        # --- training --- #
        print('-----------------------------------------')
        print('Start training at epoch %d' % epoch)
        train_iterator = data.build_iterator(batch_size=args.number_of_gpu * args.batch_size_per_gpu, mode='train')
        pbar = tqdm(enumerate(train_iterator), total=train_batch_num_per_epoch)
        epoch_step, train_loss = 0, 0.
        for _, train_batch in pbar:
            one_train_input_batch, one_train_output_batch = train_batch
            if len(one_train_input_batch) == 0 or len(one_train_output_batch) == 0:
                break
            train_batch_src_tensor, train_batch_src_mask, train_batch_input, train_batch_labels = \
                data.parse_batch_tensor(train_batch)
            if cuda_available:
                train_batch_src_tensor = train_batch_src_tensor.to(device)
                train_batch_src_mask = train_batch_src_mask.to(device)
                train_batch_input = train_batch_input.to(device)
                train_batch_labels = train_batch_labels.to(device)
            loss = model(train_batch_src_tensor, train_batch_src_mask, train_batch_input, train_batch_labels)
            loss = loss.mean()
            train_loss += loss.item()
            epoch_step += 1
            pbar.set_description(f'Epoch: {epoch}, global_step: {global_step}, '
                                 f'avg_loss: {round(train_loss / epoch_step, 2)}, '
                                 f'cur_loss: {round(loss.item(), 2)}')
            if args.gradient_accumulation_steps > 1:  # TODO verify it
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if epoch_step % args.gradient_accumulation_steps == 0 or \
                    epoch_step == train_batch_num_per_epoch:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step == actual_total_steps:
                    print('Pretraining completed at steps {}'.format(global_step))
                    break

                if global_step > 0 and global_step % args.save_steps == 0:
                    print('-----------------------------------------')
                    print('Start evaluation at global update step {}'.format(global_step))
                    model.eval()
                    # TODO change to bleu evaluation
                    # best_dev_bleu = evaluate_by_bleu(data=data, best_dev_bleu=best_dev_bleu,
                    #                                  cuda_available=cuda_available,
                    #                                  multi_gpu_training=multi_gpu_training,
                    #                                  device=device, model=model, args=args,
                    #                                  epoch=epoch, global_step=global_step)
                    best_dev_loss = evaluate_by_loss(data=data, best_dev_loss=best_dev_loss,
                                                     cuda_available=cuda_available,
                                                     multi_gpu_training=multi_gpu_training,
                                                     device=device, model=model, args=args,
                                                     epoch=epoch, global_step=global_step)
                    model.train()
                    print('dev evaluation finished.')
                    print('Resume training....')
                    print('-----------------------------------------')
        train_loss = train_loss / train_batch_num_per_epoch
        print('At epoch {}, total update steps is {}, the total training loss is {}'.format(epoch, global_step, train_loss))
        print('++++++++++++++++++++++++++++++++++++++++++')
