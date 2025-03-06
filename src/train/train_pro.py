import torch 
import os
import munch
import yaml
import sys 
from network.pro_model import ProModel
from network.pro_model import AverageValueMeter,build_optimizer,save_model_2
from time import time
from tqdm.auto import tqdm
import time as timetmp
import argparse
import logging
import random
import math
from loss.cdloss import SimplificationLoss 
from dataset_svr.trainer_dataset import build_dataset,get_spherepoints

def setFolders(args):
    LOG_DIR = args.dir_outpath
    MODEL_NAME = '%s-%s'%(args.model_name, timetmp.strftime("%m%d_%H%M", timetmp.localtime()))

    OUT_DIR = os.path.join(LOG_DIR, MODEL_NAME)
    args.dir_checkpoints = os.path.join(OUT_DIR, 'checkpoints')
    if not os.path.exists(OUT_DIR): os.mkdir(OUT_DIR)
    if not os.path.exists(args.dir_checkpoints):
        os.makedirs(args.dir_checkpoints)
    
    LOG_FOUT = open(os.path.join(OUT_DIR, 'log_%s.csv' %(MODEL_NAME)), 'w')
    return MODEL_NAME, OUT_DIR, LOG_FOUT
def log_string(out_str,LOG_FOUT):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()


def train():
    exp_name,Log_dir,LOG_FOUT = setFolders(args)
    log_string('EPOCH,avg_cd_l1,avg_cd_l2,Best CDL2[epoch,best_loss],',LOG_FOUT)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(Log_dir, 'train.log')),
                                                      logging.StreamHandler(sys.stdout)])
    logging.info(str(args))

    metrics = ['cd_p', 'cd_t', 'f1']
    best_epoch_losses = {m: (0, 0) if m == 'f1' else (0, math.inf) for m in metrics} 
    train_loss_meter = AverageValueMeter()
    val_loss_meters = {m: AverageValueMeter() for m in metrics}
    #seed
    if not args.manual_seed:
        seed = random.randint(1,10000)
    else:
        seed = int(args.manual_seed)
    logging.info('Random Seed: %d' % seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    if args.distirbuted:
        promodel = torch.nn.DataParallel(ProModel(),device_ids=args.gpus,output_device=args.gpus[0])
    else:
        promodel = ProModel.to(device)
    optimizer,scheduler = build_optimizer(promodel,args)
    best_cd_l1 = float("inf")
    best_cd_l2 = float("inf")
    best_f1 = float("inf")
    print("Data Uploading...")

    dataloader, dataloader_test = build_dataset(args)
    print("Data Preparation Done...")
    loss_function = SimplificationLoss()
    print("Loss Function Done...")
 
    for epoch in tqdm(range(args.start_epoch,args.nepoch),desc='Training'):
        # time,loss
        epoch_start_time = time()
        total_cd_l1 = 0
        total_cd_l2 = 0
        train_loss_meter.reset()

        promodel.module.train()
        
        for param_group in optimizer.param_groups:
            print(f"Epoch {epoch+1}, Learning Rate: {param_group['lr']}")
        n_batches = len(dataloader)
        # train
        with tqdm(dataloader) as t:
                for batch_idx,data in enumerate(t):
                    optimizer.zero_grad()
                    
                    n_itr = epoch * n_batches + batch_idx
                    # to(cuda)
                    images = data['image'].to(device)
                    batch_size = images.shape[0]
                    partial_pcs = torch.tensor(sphere_points).to(torch.float32).unsqueeze(0).repeat(images.shape[0], 1, 1).to(device)
                    # partial_pcs = partial_pcs.to(device)
                    pointclouds = data['points'].to(device)
                    # category
                    category = data['category']
                    category_indices = [int(c) for c in category]
                    text_labels = torch.tensor(category_indices).to(device)

                    pred_points = promodel(images,partial_pcs,text_labels)
                    # to(cuda)
                    pred_points = pred_points.to(pointclouds.device)
                    pred_points = pred_points.transpose(2,1)
                    net_loss,loss_t=loss_function(pred_points,pointclouds)

                    # caculate Chamfer distance loss            
                    net_loss = net_loss.mean()
                    loss_t = loss_t.mean()

                    # net_loss_all = net_loss + loss_t
                    net_loss_all = net_loss
                    train_loss_meter.update(net_loss.item())
                    net_loss_all.backward()
                    optimizer.step()
                
                    cd_l2_item = torch.sum(loss_t).item() / batch_size * 1e4
                    total_cd_l2 += cd_l2_item
                    cd_l1_item = net_loss.item() * 1e4
                    total_cd_l1 += cd_l1_item
                    
                    t.set_description('[Epoch %d/%d][Batch %d/%d]' % (epoch, args.nepoch, batch_idx + 1, n_batches))
                    t.set_postfix(loss='%s' % ['%.4f' % l for l in [cd_l1_item, cd_l2_item]])
        scheduler.step(epoch) # CosLR,
      
        avg_cd_l1 = total_cd_l1 / n_batches
        avg_cd_l2 = total_cd_l2 / n_batches
        epoch_end_time = time()
        logging.info('')
        logging.info(
            exp_name + '[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s ' %
            (epoch,args.nepoch,epoch_end_time - epoch_start_time,['%.4f' % l for l in [avg_cd_l1,avg_cd_l2]])
        )
        log_string(f'{epoch}[{batch_idx}],{avg_cd_l1:.4f},{avg_cd_l2:.4f},{total_cd_l2:.4f},{best_epoch_losses},', LOG_FOUT)
        if epoch % args.epoch_interval_to_val == 0 or epoch == args.nepoch - 1:
            best_cd_l1, best_cd_l2 ,best_f1= val(promodel,loss_function,epoch,val_loss_meters,dataloader_test,best_epoch_losses,LOG_FOUT,Log_dir,best_cd_l1,best_cd_l2,best_f1)

def val(net,cal_loss,curr_epoch,val_loss_meters,dataloader_test,best_epoch_losses,LOG_FOUT,log_dir,best_cd_l1,best_cd_l2,best_f1):
    val_start_time = time()
    metrics_val = ['cd_t']
    val_loss_meters = {m: AverageValueMeter() for m in metrics_val}
    logging.info('Testing...')
    for v in val_loss_meters.values():
        v.reset()
    net.module.eval()

    total_cd_l1 = 0
    total_cd_l2 = 0
    total_f1 = 0 
    n_batches = len(dataloader_test) 
    with torch.no_grad():
        with tqdm(dataloader_test) as tt:
            for i,data in enumerate(tt):
                images = data['image'].to(device)
                batch_size = images.shape[0]
                gt = data['points'].to(device)
                partial_pcs = torch.tensor(sphere_points).to(torch.float32).unsqueeze(0).repeat(images.shape[0], 1, 1).to(device)
                category = data['category']
                category_indices = [int(c) for c in category]
                text_labels = torch.tensor(category_indices).to(device)

                pred_points = net(images,partial_pcs,text_labels)
                pred_points = pred_points.transpose(2,1)
       
                loss_p, loss_t,f1 = cal_loss(pred_points, gt,calc_f1=True)

                cd_l1_item = torch.sum(loss_p).item() / batch_size * 1e4
                cd_l2_item = torch.sum(loss_t).item() / batch_size * 1e4
                f1_item = torch.sum(f1).item() / batch_size * 1e4
                total_cd_l1 += cd_l1_item
                total_cd_l2 += cd_l2_item
                total_f1 += f1_item

            avg_cd_l1 = total_cd_l1 / n_batches
            avg_cd_l2 = total_cd_l2 / n_batches
            avg_f1 = total_f1 / n_batches

            if avg_cd_l1 < best_cd_l1:
                best_cd_l1 = avg_cd_l1
                save_model_2(str(log_dir) + '/checkpoints/bestl1_network.pth', net)
                logging.info("Saving net...")
            if avg_cd_l2 < best_cd_l2:
                best_cd_l2 = avg_cd_l2
            if avg_f1 > best_f1:
                best_f1 = avg_f1

            log_string('%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f'%(curr_epoch, avg_cd_l1, best_cd_l1, avg_cd_l2, best_cd_l2,avg_f1,best_f1), LOG_FOUT)

            val_end_time = time()

            logging.info(
                '[Epoch %d/%d] TestTime = %.3f (s) Curr_cdl1 = %s Best_cdl1 = %s Curr_cdl2 = %s Best_cdl2 = %s Curr_f1 = %s Best_f1 = %s' %
                (curr_epoch, args.nepoch, val_end_time - val_start_time, avg_cd_l1, best_cd_l1, avg_cd_l2, best_cd_l2, avg_f1, best_f1))        

        return best_cd_l1, best_cd_l2 , best_f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    parser.add_argument('-gpu', '--gpu_id', help='gpu_id', required=True)
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu_id) #os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
    print('Using gpu:' + str(arg.gpu_id))
    sphere_points = get_spherepoints(args.number_points,0.5)
    device = torch.device(args.gpus[0] if torch.cuda.is_available() else 'cpu')
    print('Number of points:' + str(args.number_points))

    train()