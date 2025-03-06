import argparse
import numpy as np
import torch
import munch
# import dataset_shapenet
import dataset_svr.dataset_shapenet as dataset_shapenet
import yaml

def pc_normalize(pc, radius):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m * radius
    return pc

def get_spherepoints(num_points, radius):
    ball_name = 'Path/balls/%d.xyz' % num_points
    ball = np.loadtxt(ball_name)
    ball = pc_normalize(ball, radius)
    return ball
def build_dataset(args):
    # Create Datasets
    dataset_train = dataset_shapenet.ShapeNet(args, train=True)
    dataset_test = dataset_shapenet.ShapeNet(args, train=False)

    # Create dataloaders
    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                                    batch_size=args.batch_size,
                                                                    shuffle=True,
                                                                    num_workers=int(args.workers))
    dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                                batch_size=args.batch_size,
                                                                shuffle=False, num_workers=int(args.workers))

    len_dataset = len(dataset_train)
    len_dataset_test = len(dataset_test)
    print('Length of train dataset:%d', len_dataset)
    print('Length of test dataset:%d', len_dataset_test)

    return dataloader_train, dataloader_test

def build_dataset_val(args):

    # Create Datasets
    dataset_test = dataset_shapenet.ShapeNet_val(args, train=False)

    # Create dataloaders
    dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                                batch_size=args.batch_size,
                                                                shuffle=False, num_workers=int(args.workers))

    len_dataset_test = len(dataset_test)
    print('Length of test dataset:%d', len_dataset_test)

    return dataloader_test

if __name__ == '__main__':
    config_path = "Path/MAE/config.yaml"
    args = munch.munchify(yaml.safe_load(open(config_path)))
    dataloader_test = build_dataset_val(args)
    for data in dataloader_test:
        img = data['image']
        pc = data['points']
        print(img.shape)
        print(pc.shape)
        print('done')