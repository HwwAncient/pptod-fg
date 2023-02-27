import os
import torch

from tqdm import tqdm
from metrics import calc_bleu


def evaluate_by_loss(data, best_dev_loss, cuda_available, multi_gpu_training,
                     device, model, args, epoch, global_step):

    with torch.no_grad():
        # prepare data and generate
        dev_batch_list = data.get_batches(args.number_of_gpu * args.batch_size_per_gpu, mode='dev')
        dev_batch_num_per_epoch = len(dev_batch_list)
        dev_pbar = tqdm(dev_batch_list, total=dev_batch_num_per_epoch)
        print('Number of evaluation batches is {}'.format(dev_batch_num_per_epoch))
        dev_loss = 0.
        for one_dev_batch in dev_pbar:
            if len(one_dev_batch[0]) == 0:
                break
            dev_batch_src_tensor, dev_batch_src_mask, dev_batch_input, dev_batch_labels = \
                data.parse_batch_tensor(one_dev_batch)
            if cuda_available:
                dev_batch_src_tensor = dev_batch_src_tensor.to(device)
                dev_batch_src_mask = dev_batch_src_mask.to(device)
                dev_batch_input = dev_batch_input.to(device)
                dev_batch_labels = dev_batch_labels.to(device)
            one_dev_loss = model(dev_batch_src_tensor, dev_batch_src_mask, dev_batch_input,
                                 dev_batch_labels)
            one_dev_loss = one_dev_loss.mean()
            dev_loss += one_dev_loss.item()

        # compute metric
        dev_loss /= dev_batch_num_per_epoch
        print('current dev loss is {}, minimum dev loss is {}'.format(round(dev_loss, 2), round(best_dev_loss, 2)))

        # save
        if dev_loss < best_dev_loss:
            # saving the model with the lowest validation perplexity
            print('Saving model...')
            model_save_path = os.path.join(
                args.save_path, f'{args.save_ckpt_name}_epoch{epoch}_iter{global_step}')

            if os.path.exists(model_save_path):
                pass
            else:  # recursively construct directory
                os.makedirs(model_save_path, exist_ok=True)

            if multi_gpu_training:
                model.module.save_model(model_save_path)
            else:
                model.save_model(model_save_path)
            print(f'Model saved at {model_save_path}.')

            # removing extra checkpoints...
            # only save 1 checkpoints
            from operator import itemgetter

            fileData = {}
            test_output_dir = args.save_path
            for fname in os.listdir(test_output_dir):
                if fname.startswith(args.save_ckpt_name):
                    fileData[fname] = os.stat(os.path.join(test_output_dir, fname)).st_mtime
                else:
                    pass
            sortedFiles = sorted(fileData.items(), key=itemgetter(1))
            if len(sortedFiles) < args.max_save_num:
                pass
            else:
                delete = len(sortedFiles) - args.max_save_num
                for x in range(0, delete):
                    one_folder_name = os.path.join(test_output_dir, sortedFiles[x][0])
                    print(f'Max save num is {args.max_save_num}, delete exceeded checkpoint at {one_folder_name}!')
                    os.system('rm -r ' + one_folder_name)
            print('-----------------------------------')

        best_dev_loss = min(dev_loss, best_dev_loss)
    return best_dev_loss


def evaluate_by_bleu(data, best_dev_bleu, cuda_available, multi_gpu_training,
                     device, model, args, epoch, global_step):

    with torch.no_grad():
        # prepare data and generate
        dev_batch_list = data.get_batches(args.number_of_gpu * args.batch_size_per_gpu, mode='dev')
        dev_batch_num_per_epoch = len(dev_batch_list)
        dev_pbar = tqdm(dev_batch_list, total=dev_batch_num_per_epoch)
        print('Number of evaluation batches is {}'.format(dev_batch_num_per_epoch))
        dev_pred_text_list, dev_reference_text_list = [], []
        for one_dev_batch in dev_pbar:
            dev_batch_src_tensor, dev_batch_src_mask, dev_batch_input, dev_batch_labels = \
                data.parse_batch_tensor(one_dev_batch)
            if cuda_available:
                dev_batch_src_tensor = dev_batch_src_tensor.to(device)
                dev_batch_src_mask = dev_batch_src_mask.to(device)
                dev_batch_input = dev_batch_input.to(device)
                dev_batch_labels = dev_batch_labels.to(device)
            if multi_gpu_training:
                one_dev_prediction_text_list = model.module.batch_prediction(dev_batch_src_tensor, dev_batch_src_mask)
            else:
                one_dev_prediction_text_list = model.batch_prediction(dev_batch_src_tensor, dev_batch_src_mask)
            dev_pred_text_list += one_dev_prediction_text_list
            if multi_gpu_training:
                dev_reference_text_list += model.module.parse_batch_text(dev_batch_input)
            else:
                dev_reference_text_list += model.parse_batch_text(dev_batch_input)

        # compute metric
        assert len(dev_pred_text_list) == len(dev_reference_text_list)
        dev_bleu = calc_bleu(preds=dev_pred_text_list, targets=dev_reference_text_list)
        print('current dev bleu is {}, maximum dev bleu is {}'.format(round(dev_bleu, 4), round(best_dev_bleu, 4)))

        # TODO ---------- Begin: modify the block ------------------
        # save
        if dev_bleu > best_dev_bleu:
            # saving the model with the lowest validation bleu
            print('Saving model...')
            model_save_path = args.save_path + '/epoch_{}_dev_bleu_{}'.format(epoch, round(dev_bleu, 4))

            if os.path.exists(model_save_path):
                pass
            else:  # recursively construct directory
                os.makedirs(model_save_path, exist_ok=True)

            if multi_gpu_training:
                model.module.save_model(model_save_path)
            else:
                model.save_model(model_save_path)
            print(f'Model saved at {model_save_path}.')

            # save predicted files
            dev_pred_save_path = model_save_path + '/dev_predicted_result.txt'
            with open(dev_pred_save_path, 'w', encoding='utf8') as o:
                for text in dev_pred_text_list:
                    o.writelines(text + '\n')
            dev_reference_save_path = model_save_path + '/dev_reference_result.txt'
            with open(dev_reference_save_path, 'w', encoding='utf8') as o:
                for text in dev_reference_text_list:
                    o.writelines(text + '\n')

            # removing extra checkpoints...
            from operator import itemgetter

            fileData = {}
            test_output_dir = args.save_path
            for fname in os.listdir(test_output_dir):
                if fname.startswith('epoch'):
                    fileData[fname] = os.stat(os.path.join(test_output_dir, fname)).st_mtime
                else:
                    pass
            sortedFiles = sorted(fileData.items(), key=itemgetter(1))
            if len(sortedFiles) < args.max_save_num:
                pass
            else:
                delete = len(sortedFiles) - args.max_save_num
                for x in range(0, delete):
                    one_folder_name = os.path.join(test_output_dir, sortedFiles[x][0])
                    print(f'Max save num is {args.max_save_num}, delete exceeded checkpoint at {one_folder_name}!')
                    os.system('rm -r ' + one_folder_name)
            print('-----------------------------------')
        # TODO ---------- End: modify the block --------------------

        best_dev_bleu = max(dev_bleu, best_dev_bleu)
    return best_dev_bleu
