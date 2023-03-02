import os
import torch
import argparse
import random
import math
import numpy as np
import torch.distributed as dist

from tqdm import tqdm
from dataset import LazyDataset
from dataclass import PRETRAINING_CORPUS
from utils import torch_distributed_zero_first, print_rank_0
from evaluation import evaluate_by_loss, evaluate_by_bleu
from modelling.T5Model import T5Gen_Model
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup


def parse_config():
    parser = argparse.ArgumentParser()

    # data specification
    parser.add_argument('--dataset_prefix_path', type=str, help='The path where all datasets are stored.')
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
    parser.add_argument("--warmup_steps", default=-1, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size_per_gpu", type=int, default=4, help='Batch size for each gpu.')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="gradient accumulation step.")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_path", type=str, help="Directory to save the model parameters.")
    parser.add_argument("--save_ckpt_name", type=str, help="The name under which to save the pre-trained model. small or base or large")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()

    # setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        args.device = torch.device("cpu")
        args.world_size = 1
        args.rank = 0
        args.n_gpu = torch.cuda.device_count()
    else:  # initializes the distributed backend which will take care of synchronizing nodes/GPUs
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()
        args.n_gpu = 1
    print(f"[Distributed info]: world_size {args.world_size}, rank {args.rank}, local rank {args.local_rank}.")

    def set_seed(seed):
        """ fix random seed """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if args.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

    def create_dataloader(mode="train"):
        """ create dataloader """
        data_file = os.path.join(args.dataset_prefix_path, f"{mode}.json")
        assert os.path.exists(data_file), f"{data_file} isn't exist"
        with torch_distributed_zero_first(rank=args.local_rank):
            dataset = LazyDataset(data_file)

        if dist.is_initialized():
            data_sampler = DistributedSampler(dataset, shuffle=(mode == "train"))
            data_loader = DataLoader(dataset, batch_size=args.batch_size_per_gpu,
                                     sampler=data_sampler, collate_fn=data.collate_fn)
        else:
            data_loader = DataLoader(dataset, batch_size=args.batch_size_per_gpu,
                                     shuffle=(mode == "train"), collate_fn=data.collate_fn)
        return data_loader

    # set seed
    set_seed(seed=args.seed)

    # set reader
    print_rank_0('Start loading data...')
    preprocessed_tokenizer_path = os.path.join(args.dataset_prefix_path, 'tokenizer_with_special_token')
    with torch_distributed_zero_first(rank=args.local_rank):
        tokenizer = BertTokenizer.from_pretrained(preprocessed_tokenizer_path)
        data = PRETRAINING_CORPUS(tokenizer=tokenizer, dataset_prefix_path=args.dataset_prefix_path,
                                  max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len)

    # load data
    train_loader = create_dataloader(mode='train')
    dev_loader = create_dataloader(mode='dev')
    test_loader = create_dataloader(mode='test')
    print_rank_0('Data Loaded.')

    # load model
    print_rank_0('Start loading model...')
    with torch_distributed_zero_first(rank=args.local_rank):
        model = T5Gen_Model(model_path=args.model_name, tokenizer=tokenizer, data=data, dropout=args.dropout)
    model.to(args.device)
    if dist.is_initialized():
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    print_rank_0('Model loaded.')

    def set_optimizers(num_training_steps_per_epoch):
        """
        set the optimizer and the learning rate scheduler.
        """
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        num_training_steps = num_training_steps_per_epoch * args.num_train_epochs
        num_warmup_steps = args.warmup_steps if args.warmup_steps >= 0 else int(num_training_steps * 0.1)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps)

        # log info
        print_rank_0("***** Running training *****")
        print_rank_0(f"  Num Training steps per epoch = {num_training_steps_per_epoch}")
        print_rank_0(f"  Num Epochs = {args.num_train_epochs}")
        print_rank_0(f"  Batch size  = {args.batch_size_per_gpu}")
        print_rank_0(f"  Total optimization steps = {num_training_steps}")
        print_rank_0(f"  Total warmup steps = {num_warmup_steps}")
        print_rank_0(f"  Distributed info: world_size {args.world_size}, rank {args.rank}, local rank {args.local_rank}.")
        print_rank_0("*****************************")
        return optimizer, scheduler

    # organize optimizer
    num_training_batches_per_epoch = len(train_loader)
    num_training_steps_per_epoch = math.ceil(num_training_batches_per_epoch / args.gradient_accumulation_steps)
    optimizer, scheduler = set_optimizers(num_training_steps_per_epoch=num_training_steps_per_epoch)
    optimizer.zero_grad()

    # --- training --- #
    global_step, best_dev_loss, best_dev_bleu = 0, 1e10, 0
    for epoch in range(1, args.num_train_epochs + 1):
        print_rank_0('-----------------------------------------')
        print_rank_0('Start training at epoch %d' % epoch)
        model.train()

        if dist.is_initialized():  # make shuffling work properly across multiple epochs
            assert isinstance(train_loader.sampler, DistributedSampler)
            train_loader.sampler.set_epoch(epoch)
        if args.local_rank == -1 or args.rank == 0:
            train_pbar = tqdm(train_loader, total=num_training_batches_per_epoch)
        else:
            train_pbar = train_loader

        epoch_step, train_loss = 0, 0.
        for train_batch in train_pbar:
            train_batch = type(train_batch)(map(lambda item: item.to(args.device), train_batch))
            loss = model(*train_batch)
            train_loss += loss.item()
            epoch_step += 1

            if args.local_rank == -1 or args.rank == 0:
                train_pbar.set_description(f'Epoch: {epoch}, global_step: {global_step}, '
                                           f'avg_loss: {round(train_loss / epoch_step, 2)}, '
                                           f'cur_loss: {round(loss.item(), 2)}')

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if epoch_step % args.gradient_accumulation_steps == 0 or \
                    epoch_step == num_training_batches_per_epoch:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step > 0 and global_step % args.save_steps == 0:
                    print_rank_0('-----------------------------------------')
                    print_rank_0('Start evaluation at global update step {}'.format(global_step))

                    model.eval()
                    # TODO add individual bleu evaluation
                    best_dev_bleu = evaluate_by_bleu(data_loader=dev_loader, best_dev_bleu=best_dev_bleu,
                                                     model=model, args=args, epoch=epoch, global_step=global_step)
                    # best_dev_loss = evaluate_by_loss(data_loader=dev_loader, best_dev_loss=best_dev_loss,
                    #                                  model=model, args=args, epoch=epoch, global_step=global_step)

                    model.train()

                    print_rank_0('dev evaluation finished.')
                    print_rank_0('Resume training....')
                    print_rank_0('-----------------------------------------')

        train_loss = train_loss / num_training_batches_per_epoch
        print_rank_0('At epoch {}, total update steps is {}, the total training loss is {}'.format(epoch, global_step, train_loss))
        print_rank_0('++++++++++++++++++++++++++++++++++++++++++')
