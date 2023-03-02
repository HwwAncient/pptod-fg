import os
import torch
import torch.distributed as dist

from tqdm import tqdm
from metrics import calc_bleu
from operator import itemgetter
from utils import print_rank_0, gather_across_procs, reduce_mean_across_procs


def save(model, model_save_path):
    print_rank_0('Saving model...')
    if os.path.exists(model_save_path):
        pass
    else:  # recursively construct directory
        os.makedirs(model_save_path, exist_ok=True)
    if dist.is_initialized():
        model.module.save_model(model_save_path)
    else:
        model.save_model(model_save_path)
    print_rank_0(f'Model saved at {model_save_path}.')


def save_file(text_list, file_save_path):
    with open(file_save_path, 'w', encoding='utf8') as o:
        for text in text_list:
            o.writelines(text + '\n')


def remove_extra_files(args):
    fileData = {}
    test_output_dir = args.save_path
    for file_name in os.listdir(test_output_dir):
        if file_name.startswith(args.save_ckpt_name):
            fileData[file_name] = os.stat(os.path.join(test_output_dir, file_name)).st_mtime
        else:
            pass
    sortedFiles = sorted(fileData.items(), key=itemgetter(1))
    if len(sortedFiles) < args.max_save_num:
        pass
    else:
        delete = len(sortedFiles) - args.max_save_num
        for x in range(0, delete):
            one_folder_name = os.path.join(test_output_dir, sortedFiles[x][0])
            print_rank_0(f'Max save num is {args.max_save_num}, delete exceeded checkpoint at {one_folder_name}!')
            os.system('rm -r ' + one_folder_name)
    print_rank_0('-----------------------------------')


def evaluate_by_loss(data_loader, best_dev_loss, model, args, epoch, global_step):

    with torch.no_grad():
        # prepare data and generate
        num_dev_batches_per_epoch = len(data_loader)
        if args.local_rank == -1 or args.rank == 0:
            dev_pbar = tqdm(data_loader, total=num_dev_batches_per_epoch)
        else:
            dev_pbar = data_loader
        print_rank_0('Number of evaluation batches is {}'.format(num_dev_batches_per_epoch))

        dev_loss = 0.
        for dev_batch in dev_pbar:
            dev_batch = type(dev_batch)(map(lambda item: item.to(args.device), dev_batch))
            loss = model(*dev_batch)
            if dist.is_initialized():
                loss = reduce_mean_across_procs(tensor=loss, world_size=args.world_size)
            dev_loss += loss.item()

        # compute metric
        dev_loss /= num_dev_batches_per_epoch
        print_rank_0('current dev loss is {}, minimum dev loss is {}'.format(round(dev_loss, 2), round(best_dev_loss, 2)))

        # save
        if (dev_loss < best_dev_loss) and (args.local_rank == -1 or args.rank == 0):
            # saving the model with the lowest validation perplexity
            model_save_path = os.path.join(args.save_path, f'{args.save_ckpt_name}_epoch{epoch}_iter{global_step}')
            save(model=model, model_save_path=model_save_path)
            # removing extra checkpoints
            remove_extra_files(args=args)

        best_dev_loss = min(dev_loss, best_dev_loss)
    return best_dev_loss


def evaluate_by_bleu(data_loader, best_dev_bleu, model, args, epoch, global_step):

    with torch.no_grad():
        # prepare data and generate
        num_dev_batches_per_epoch = len(data_loader)
        if args.local_rank == -1 or args.rank == 0:
            dev_pbar = tqdm(data_loader, total=num_dev_batches_per_epoch)
        else:
            dev_pbar = data_loader
        print_rank_0('Number of evaluation batches is {}'.format(num_dev_batches_per_epoch))

        dev_pred_text_list, dev_refer_text_list = [], []
        for dev_batch in dev_pbar:
            batch_prediction = model.module.batch_prediction if dist.is_initialized() else model.batch_prediction
            parse_batch_text = model.module.parse_batch_text if dist.is_initialized() else model.parse_batch_text

            dev_batch = type(dev_batch)(map(lambda item: item.to(args.device), dev_batch))
            dev_batch_src_tensor, dev_batch_src_mask, dev_batch_input, dev_batch_labels = dev_batch
            dev_batch_pred = batch_prediction(dev_batch_src_tensor, dev_batch_src_mask, return_tensor=True)

            if dist.is_initialized():
                dev_batch_pred = gather_across_procs(tensor=dev_batch_pred, world_size=args.world_size,
                                                     max_len=model.module.max_tgt_len,
                                                     padding_value=model.module.pad_token_id)
                dev_batch_input = gather_across_procs(tensor=dev_batch_input, world_size=args.world_size,
                                                      max_len=model.module.max_tgt_len,
                                                      padding_value=model.module.pad_token_id)

            dev_pred_text_list += parse_batch_text(dev_batch_pred)
            dev_refer_text_list += parse_batch_text(dev_batch_input)

        # compute metric
        assert len(dev_pred_text_list) == len(dev_refer_text_list)
        dev_bleu = calc_bleu(preds=dev_pred_text_list, targets=dev_refer_text_list)
        print_rank_0('current dev bleu is {}, maximum dev bleu is {}'.format(round(dev_bleu, 4), round(best_dev_bleu, 4)))

        # save
        if (dev_bleu > best_dev_bleu) and (args.local_rank == -1 or args.rank == 0):
            # saving the model with the lowest validation bleu
            model_save_path = os.path.join(args.save_path, f'{args.save_ckpt_name}_epoch{epoch}_iter{global_step}')
            save(model=model, model_save_path=model_save_path)
            # saving predicted files
            dev_pred_save_path = os.path.join(model_save_path, 'dev_predicted_result.txt')
            save_file(text_list=dev_pred_text_list, file_save_path=dev_pred_save_path)
            dev_refer_save_path = os.path.join(model_save_path, 'dev_reference_result.txt')
            save_file(text_list=dev_refer_text_list, file_save_path=dev_refer_save_path)
            # removing extra checkpoints
            remove_extra_files(args=args)

        best_dev_bleu = max(dev_bleu, best_dev_bleu)
    return best_dev_bleu
