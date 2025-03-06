import torch 
import argparse
import os
import random
import logging
from dataset_svr.trainer_dataset import build_dataset_val,get_spherepoints
from network.pro_model import ProModel
from network.pro_model import AverageValueMeter
from loss.cdloss import SimplificationLoss 
from tqdm import tqdm
import munch
import yaml
import os 

def val():
    device = torch.device(args.gpus[0] if torch.cuda.is_available() else 'cpu')
    dataloader_test = build_dataset_val(args)
    len_test = len(dataloader_test)

    if not args.manual_seed:
        seed = random.randint(1,10000)
    else:
        seed = int(args.manual_seed)
    logging.info('Random Seed: %d' % seed)
    random.seed(seed)
    torch.manual_seed(seed)
    ckpt = torch.load(args.ckpt_path,map_location=device)
    if args.distributed:
        promodel = torch.nn.DataParallel(ProModel(),device_ids=args.gpus,output_device=args.gpus[0])
        promodel.to(device)
        promodel.module.load_state_dict(ckpt['net_state_dict'])
        logging.info("%s's previous weights loaded." % args.model_name)
        promodel.module.eval()
    else:
        promodel = ProModel().to(device)
        promodel.load_state_dict(ckpt['net_state_dict'])
        logging.info("%s's previous weights loaded." % args.model_name)
        promodel.eval()
    
    logging.info('Testing....')

    test_loss_l1 = AverageValueMeter()
    test_loss_l2 = AverageValueMeter()
    test_f1 = AverageValueMeter()
    loss_function = SimplificationLoss()
    sphere_points = get_spherepoints(args.number_points,0.5)


    with tqdm(dataloader_test) as t:
        for i,data in enumerate(t):
            with torch.no_grad():
                images = data['image'].to(device)
                gt = data['points'].to(device)
                partial_pcs = torch.tensor(sphere_points).to(torch.float32).unsqueeze(0).repeat(images.shape[0], 1, 1).to(device)
                
                category = data['category']
                category_indices = [int(c) for c in category]
                text_labels = torch.tensor(category_indices).to(device)

                batch_size = gt.shape[0]
    
                pred_points = promodel(images,partial_pcs,text_labels)
                pred_points = pred_points.transpose(2,1)
                loss_p,loss_t,f1 = loss_function(pred_points,gt,calc_f1=True)

                cd_l1_item = torch.sum(loss_p).item() / batch_size
                cd_l2_item = torch.sum(loss_t).item() / batch_size
                f1_item = torch.sum(f1).item() / batch_size
                test_loss_l1.update(cd_l1_item, images.shape[0])
                test_loss_l2.update(cd_l2_item, images.shape[0])
                test_f1.update(f1_item, images.shape[0])

    print('cd_l1 %f cd_l2 %f f1 %f' % (test_loss_l1.avg, test_loss_l2.avg,test_f1.avg))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    parser.add_argument('-gpu', '--gpu_id', help='gpu_id', required=True)
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu_id)
    print('Using gpu:' + str(arg.gpu_id))
    val()
    val()