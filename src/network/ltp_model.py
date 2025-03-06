import munch
import torch
import torch.nn as nn
# from tools import builder
import torch.optim as optim
import yaml
from timm.scheduler import CosineLRScheduler
from models.ULIP_models import ULIP_PointBERT
from models.text_encoder_3d import CLIPTextEncoder
from models.tokenizer import SimpleTokenizer
import numpy as np
import shutil
import torch.distributed as dist
from collections import OrderedDict

def get_prompt_text(category: list, label_map: dict) -> list:
    prompt_text = []
    for cat in category:
        cat_str = str(cat)
        if cat_str in label_map:
            prompt_text.append(label_map[cat_str])
        else:
            prompt_text.append('Unknown') 
    return prompt_text

class TextAlignPCModel(nn.Module):
    def __init__(self,
        args,
        device = "cuda:0",
        ):
            super(TextAlignPCModel, self).__init__()

            self.device = device
            self.prompt_prefix = 10  
            self.prompt_postfix = 10  
            self.text_prompt_embeddings = nn.Embedding(77,512)
            nn.init.normal_(self.text_prompt_embeddings.weight, std=0.02)
            #TEXT ENCODER
            self.text_encoder = CLIPTextEncoder(pretrained=args.clip_ckpt_path)
            self.text_encoder.init_weights()
            self.exclude_key = 'prompt'
            for n,m in self.text_encoder.named_parameters():
                if self.exclude_key not in n:
                    m.requires_grad = False
                else:
                    continue
            #tokenizer
            self.tokenizer = SimpleTokenizer()
            #pc encoder
            config_path = args.ulip
            args = munch.munchify(yaml.safe_load(open(config_path)))
            self.ULIP = ULIP_PointBERT(args)
            ckpt = torch.load(args.ulip_ckpt_path)
            state_dict = OrderedDict()
            for k, v in ckpt["state_dict"].items():
                # print(f'k {k}')
                state_dict[k.replace("module.", "")] = v
            self.ULIP.load_state_dict(state_dict, strict=False)
            for ulip_param in self.ULIP.parameters():
                ulip_param.requires_grad = False

            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            #map
            self.label_map = dict({'2691156':'airplane','2828884':'bench','2933112':'cabinet','2958343':'car','3001627':'chair',
                         '3211117':'display','3636649':'lamp','3691459':'loudspeaker','4090263':'rifle',
                         '4256520':'sofa','4379243':'table','4401088':'telephone','4530566':'vessel'})
    def encode_textprompt(self, text):
        word_tokens = self.tokenizer(text).to(self.device)
        word_embedding = self.text_encoder.encode_token(word_tokens)
        text_embeddings = self.text_prompt_embeddings(torch.arange(77).to(self.device)).unsqueeze(0).repeat([len(text), 1, 1])
        text_tokens = torch.zeros(len(text), 77).to(self.device)
        
        for i in range(len(text)):
            ind = torch.argmax(word_tokens[i], -1)
            print(f"ind:{ind}")
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = word_embedding[i, 1: ind]
            text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = word_embedding[i, ind]
            text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = word_tokens[i, ind]

        text_features = self.text_encoder(text_embeddings, text_tokens)

        return text_features  
    def forward(self,images,text_labels,pc):
        B,_,_ = pc.shape
        
        text_labels = text_labels.tolist()
        outputs = self.ULIP(pc)
        pc_embed = outputs['pc_embed']
        image_embed = torch.randn(1,2).to(pc.device)

        prompt_text = get_prompt_text(text_labels,self.label_map)
        text_embed = self.encode_textprompt(prompt_text).to(torch.float32)

      
        return {'text_embed': text_embed,
                'pc_embed': pc_embed,
                'image_embed': image_embed,
                'logit_scale': self.logit_scale.exp()}

class AverageValueMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def build_optimizer(base_model,config):
    opti_config = config.optimizer
    if opti_config.type == 'AdamW':
        def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
            decay = []
            no_decay = []
            for name, param in model.module.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                    # print(name)
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]

        param_groups = add_weight_decay(base_model, weight_decay=opti_config.kwargs.weight_decay)
        optimizer = optim.AdamW(param_groups, **opti_config.kwargs)
    elif opti_config.type == 'Adam':
        optimizer = optim.Adam(base_model.parameters(), **opti_config.kwargs)
    elif opti_config.type == 'SGD':
        optimizer = optim.SGD(base_model.parameters(), nesterov=True, **opti_config.kwargs)
    else:
        raise NotImplementedError()
    sche_config = config.scheduler
    if sche_config.type == 'CosLR':
        scheduler = CosineLRScheduler(optimizer,
                                      t_initial=sche_config.kwargs.epochs,
                                    #   t_mul=1,
                                      lr_min=1e-7,
                                    #   decay_rate=0.1,
                                      warmup_lr_init=1e-6,
                                      warmup_t=sche_config.kwargs.initial_epochs,
                                      cycle_limit=1,
                                      t_in_epochs=True)
        return optimizer,scheduler
    elif sche_config.type == 'LambdaLR':
        kwargs = sche_config.kwargs
        lr_lbmd = lambda e: max(kwargs.lr_decay ** (e / kwargs.decay_step), kwargs.lowest_decay)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer,lr_lbmd)
        return optimizer,scheduler
    return optimizer 

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def get_model(model):
    if isinstance(model, torch.nn.DataParallel) \
      or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def synchronize(self):
        for meter in self.meters:
            meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def  build_optimizer(base_model,config):
    opti_config = config.optimizer
    if opti_config.type == 'AdamW':
        def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
            decay = []
            no_decay = []
            for name, param in model.module.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                    # print(name)
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]

        param_groups = add_weight_decay(base_model, weight_decay=opti_config.kwargs.weight_decay)
        optimizer = optim.AdamW(param_groups, **opti_config.kwargs)
    elif opti_config.type == 'Adam':
        optimizer = optim.Adam(base_model.parameters(), **opti_config.kwargs) 
    elif opti_config.type == 'SGD':
        optimizer = optim.SGD(base_model.parameters(), nesterov=True, **opti_config.kwargs)
    else:
        raise NotImplementedError()
    sche_config = config.scheduler
    if sche_config.type == 'CosLR':
        scheduler = CosineLRScheduler(optimizer,
                                      t_initial=sche_config.kwargs.epochs,
                                    #   t_mul=1,
                                      lr_min=1e-7,
                                    #   decay_rate=0.1,
                                      warmup_lr_init=1e-5,
                                      warmup_t=sche_config.kwargs.initial_epochs,
                                      cycle_limit=1,
                                      t_in_epochs=True)
        return optimizer,scheduler
    elif sche_config.type == 'LambdaLR':
        kwargs = sche_config.kwargs
        lr_lbmd = lambda e: max(kwargs.lr_decay ** (e / kwargs.decay_step), kwargs.lowest_decay)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer,lr_lbmd)
        return optimizer,scheduler
    return optimizer 
def scaled_all_reduce(tensors, is_scale=True):
    """Performs the scaled all_reduce operation on the provided tensors.
    The input tensors are modified in-place. Currently supports only the sum
    reduction operator. The reduced values are scaled by the inverse size of the
    world size.
    """
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    # Queue the reductions
    reductions = []
    for tensor in tensors:
        reduction = dist.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    # Wait for reductions to finish
    for reduction in reductions:
        reduction.wait()
    # Scale the results
    if is_scale:
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors
def save_on_master(state, is_best, output_dir):
    if is_main_process():
        ckpt_path = '{}/checkpoint_{}.pt'.format(output_dir, state['epoch'])
        best_path = f'{output_dir}/checkpoint_best.pt'
        torch.save(state, ckpt_path)
        if is_best:
            shutil.copyfile(ckpt_path, best_path)

def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(
            tensor_all,
            tensor,
            async_op=False  # performance opt
        )

        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor
if __name__ =='__main__':
    print("Hello world")
    # ULIP 初始化代码
    import argparse
    import torch
    import munch
    import yaml
    # device = 'cuda:0'
    # # config_path = "Path/cfgs/ULIP.yaml"
    # # args = munch.munchify(yaml.safe_load(open(config_path)))
    
    # Alion = TextAlignPCModel().to(device)
    # # pc
    # pc = torch.randn(2,2048,3).to(device)
    # # text
    # text = ['02691156','02828884']
    # category_indices = [int(c) for c in text]
    # text_labels = torch.tensor(category_indices).to(device)
    # # text_labels = text_labels.unsqueeze(0).to(device)
    # # image
    # image = torch.randn(2,3,224,224).to(device)
    # # inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in inputs]
    # outputs = Alion(image,text_labels,pc)
    # pc_embed = outputs['pc_embed']
    # text_embed = outputs['text_embed']
    # # print(f'outputs: {outputs}')
    # print(f'pc shape: {pc_embed.shape}')
    # print(f'text shape: {text_embed.shape}')
