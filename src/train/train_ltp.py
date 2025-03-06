from tqdm import tqdm
from dataset_svr.trainer_text_dataset import *
import torch
import torch.nn.functional as F
from models.ULIP_models import get_loss_v2
import sys
import time
from network.ltp_model import *
from network.ltp_model import TextAlignPCModel
from models.ULIP_utils import *
from dataset_svr.trainer_dataset import build_dataset
import os
import logging
import time as timetmp


def setFolders(args):
    LOG_DIR = args.dir_outpath
    MODEL_NAME = '%s-%s'%(args.model, timetmp.strftime("%m%d_%H%M", timetmp.localtime()))

    OUT_DIR = os.path.join(LOG_DIR, MODEL_NAME)
    args.dir_checkpoints = os.path.join(OUT_DIR, 'checkpoints')
    if not os.path.exists(OUT_DIR): os.mkdir(OUT_DIR)
    if not os.path.exists(args.dir_checkpoints):
        os.makedirs(args.dir_checkpoints)
    
    LOG_FOUT = open(os.path.join(OUT_DIR, 'log_%s.csv' %(MODEL_NAME)), 'w')
    return MODEL_NAME, OUT_DIR, args.dir_checkpoints
def log_string(out_str,LOG_FOUT):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
def save_model(path, net, net_d=None):
    if net_d is not None:
        torch.save({'net_state_dict': net.state_dict(),
                    'D_state_dict': net_d.state_dict()}, path)
    else:
        torch.save({'net_state_dict': net.state_dict()}, path)
def main(args):
    best_cos = float(0)
    exp_name,Log_dir,Check_FOUT = setFolders(args)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(Log_dir, 'train.log')),
                                                      logging.StreamHandler(sys.stdout)])
    logging.info(str(args))
    dataloader_train, dataloader_test = build_dataset(args)
    # create model
    print("=> creating model: {}".format(args.model))
    model = TextAlignPCModel(args).to(device)

    criterion = get_loss_v2(args).to(device)
    optimizer,scheduler = build_optimizer(model,args)
    print("=> beginning training")
    best_epoch = -1
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train(dataloader_train,model,criterion,optimizer,epoch,scheduler,args)
        logging.info('')
        logging.info(f'Training result: {train_stats}')
        logging.info('Testing...')
        val_stats = {"cos@t&p":-1}
        
        if epoch % 1 == 0:
            val_stats = val(dataloader_test,model,args)
            cos = val_stats["cos@t&p"]
            print(f'val_stats:{val_stats}')

            is_best = cos > best_cos
            if is_best:
                best_epoch = epoch
            
            best_cos = max(cos,best_cos)
            if is_best :
                print("=> saving checkpoint")
                save_model(str(Check_FOUT)+'/prompt.pth',model.text_prompt_embeddings)
                save_model(str(Check_FOUT)+'/text_encoder.pth',model.text_encoder)
                logging.info("Saving net...")
            if epoch + 1 == args.epochs:
                print("=> saving last checkpoint")
                save_model(str(Check_FOUT)+'/last_prompt.pth',model.text_prompt_embeddings)
                save_model(str(Check_FOUT)+'/last_text_encoder.pth',model.text_encoder)
 
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch,
                     'best_cos@t&p': best_cos,
                     'best_epoch': best_epoch}
        logging.info(f'log_stats: {log_stats}')
 
def train(train_loader, model, criterion, optimizer, epoch, scheduler, args):
        n_batches = len(train_loader)
        # switch to train mode
        total_loss = 0
        model.train()
        end = time.time()
        with tqdm(train_loader) as t:
            for data_iter,data in enumerate(t):
                optimizer.zero_grad()
                # images 
                images = data['image'].to(device)
                # text  
                category = data['category']
                category_indices = [int(c) for c in category]
                text_labels = torch.tensor(category_indices).to(device)
                # pointclouds
                pc = data['points'].to(device)

                outputs = model(images,text_labels,pc)
                loss_dict = criterion(outputs)
                loss = loss_dict['loss']

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                get_model(model).logit_scale.data.clamp_(0, 4.6052)
                logit_scale = get_model(model).logit_scale.exp().item()
        avg_loss = total_loss / n_batches
        scheduler.step(epoch)
        return {**{'loss':avg_loss},
                    'lr': optimizer.param_groups[0]['lr'],
                    'logit_scale': logit_scale}
def val(test_loader,model,tokenizer,args=None):
    batch_time = AverageMeter('Time', ':6.3f')
    total = 0
    avg = 0
    model.eval()
    n_batches = len(test_loader) 
    
    print('==>encoding captions')
    with torch.no_grad():
        end = time.time()
        for i,data in enumerate(test_loader):
            # image 
            images = data['image'].to(device)
            batch_size = images.shape[0]
            # category
            category = data['category']
            category_indices = [int(c) for c in category]
            text_labels = torch.tensor(category_indices).to(device)
            # pointclouds
            gt = data['points'].to(device)
            # model
            outputs = model(images,text_labels,gt)
            text_embed = outputs['text_embed']
            gt_embed = outputs['pc_embed']

            text_embed = F.normalize(text_embed, dim=-1, p=2)
            gt_embed = F.normalize(gt_embed, dim=-1, p=2)

            logits_per_pc_text =  text_embed @ gt_embed.t()
    
            cos_sim_1 = torch.diag(logits_per_pc_text).mean()
        
            total += cos_sim_1
    
    avg = total / n_batches

    batch_time.update(time.time() - end)
    end = time.time()
    logging.info(f"batch_time {batch_time}, 'cos@t&p':{avg}")
    return {'cos@t&p':avg}

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    parser.add_argument('-gpu', '--gpu_id', help='gpu_id', required=True)
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))
    device = torch.device(args.gpu_id if torch.cuda.is_available() else 'cpu')
    print('Using gpu:' + str(arg.gpu_id))

    train()