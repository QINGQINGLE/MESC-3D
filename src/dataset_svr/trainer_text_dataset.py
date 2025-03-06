import argparse
import torch
import munch
import dataset_svr.dataset_shapenet_text as dataset_shapenet
import yaml
import numpy as np
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
def get_batch_label(texts, prompt_text, label_map: dict):
    label_vectors = torch.zeros(0)
    if len(label_map) != 7:
        if len(label_map) == 2:
            for text in texts:
                label_vector = torch.zeros(2)
                if text == 'Normal':
                    label_vector[0] = 1
                else:
                    label_vector[1] = 1
                label_vector = label_vector.unsqueeze(0)
                label_vectors = torch.cat([label_vectors, label_vector], dim=0)
        else:
            for text in texts:
                label_vector = torch.zeros(len(prompt_text))
                if text in label_map:
                    label_text = label_map[text]
                    label_vector[prompt_text.index(label_text)] = 1

                label_vector = label_vector.unsqueeze(0)
                label_vectors = torch.cat([label_vectors, label_vector], dim=0)
    else:
        for text in texts:
            label_vector = torch.zeros(len(prompt_text))
            labels = text.split('-')
            for label in labels:
                if label in label_map:
                    label_text = label_map[label]
                    label_vector[prompt_text.index(label_text)] = 1
            
            label_vector = label_vector.unsqueeze(0)
            label_vectors = torch.cat([label_vectors, label_vector], dim=0)

    return label_vectors
# def get_prompt_text(category:list,label_map: dict):
#     prompt_text = []
#     for v in label_map.values():
#         prompt_text.append(v)

#     return prompt_text
def get_prompt_text(category: list, label_map: dict) -> list:
    prompt_text = []
    for cat in category:
        if cat in label_map:
            prompt_text.append(label_map[cat])
        else:
  
            prompt_text.append('Unknown') 
    return prompt_text
if __name__ == '__main__':
    from tqdm import tqdm
    from models.tokenizer import SimpleTokenizer
    from models.ULIP_models import ULIP_PointBERT
    from collections import OrderedDict
    config_path = 'Path/MAE/config.yaml'
    args = munch.munchify(yaml.safe_load(open(config_path)))
    dataloader_train, dataloader_test = build_dataset(args)
    tokenizer = SimpleTokenizer()
    label_map = dict({'02691156':'airplane','02828884':'bench','02933112':'cabinet','02958343':'car','03001627':'chair',
                         '03211117':'display','03636649':'lamp','03691459':'loudspeaker','04090263':'rifle',
                         '04256520':'sofa','04379243':'table','04401088':'telephone','04530566':'vessel'})
    # prompt_text = get_prompt_text(label_map)
    # print(f'prompt_text:{prompt_text}')
    config_path = "Path/cfgs/ULIP.yaml"
    args = munch.munchify(yaml.safe_load(open(config_path)))
    ULIP = ULIP_PointBERT(args).to('cuda:7')
    ckpt = torch.load('Path/checkpoints/pretrained_models_ckpt_zero-sho_classification_checkpoint_pointbert.pt')
    state_dict = OrderedDict()
    for k, v in ckpt["state_dict"].items():
        # print(f'k {k}')
        state_dict[k.replace("module.", "")] = v
    ULIP.load_state_dict(state_dict, strict=False)
    tokenized_captions = []
    with tqdm(dataloader_train) as t:
        for batch_idx,data in enumerate(t):
            # if batch_idx == 0:
                # category
            tokenized_captions = []
            category = data['category']
            text_labels = list(category)
            # text_labels = get_batch_label(text_labels, prompt_text, label_map)
            prompt_text = get_prompt_text(text_labels,label_map)
            path = data['image_path']
            # captions 
            caption = data["caption"]
            print(f'caption : {caption}')
            caption = list(zip(*caption))
            for i in range(len(caption)):
                caption[i] = list(caption[i])
            for i in range(len(caption)):
                texts = tokenizer(caption[i]).cuda(7, non_blocking=True)
                tokenized_captions.append(texts)
            tokenized_captions = torch.stack(tokenized_captions)
            text_features = []
            for i in range(tokenized_captions.shape[0]):
                text_for_one_sample = tokenized_captions[i]
                text_embed = ULIP.encode_text(text_for_one_sample)
                text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
                text_embed = text_embed.mean(dim=0)
                text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
                text_features.append(text_embed)
            text_features = torch.stack(text_features)    
            
            # print(f'tokenized {tokenized_captions.shape}')
            # print(f"catefory {category}")
            # print(f"text_labels {text_labels}")
            # print(f'prompt_text:{prompt_text}')
            print(f'text_embed {text_features.shape}')
                # print(f'path {path}')
            