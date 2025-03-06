import munch
import torch
import torch.nn as nn
from tools import builder
import torch.optim as optim
import yaml
from timm.scheduler import CosineLRScheduler
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
# from dataset_svr.trainer_dataset import get_spherepoints

print("Hello World")

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=20,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  

        attn = attn.softmax(dim=-1)
        weights = attn

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x, weights
    
class CrossAttention(nn.Module):
    def __init__(self,dim,out_dim,num_heads=8,qkv_bias=False,qk_scale=None,
                 attn_drop=0.0,proj_drop=0.0,):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads 
        self.scale = qk_scale or head_dim **-0.5

        self.q = nn.Linear(dim,out_dim,bias=qkv_bias) 
        self.k = nn.Linear(dim,out_dim,bias=qkv_bias)
        self.v = nn.Linear(dim,out_dim,bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(out_dim,out_dim)
        self.proj_drop =nn.Dropout(proj_drop)

    def forward(self,q,v):
        B,N,_ = q.shape 
        C = self.out_dim 
        k = v 
        NK = k.size(1)

        q = (
            self.q(q)
            .view(B,N,self.num_heads,C // self.num_heads)
            .permute(0,2,1,3)
        ) 
        k = (
            self.k(k)
            .view(B,NK,self.num_heads,C // self.num_heads)
            .permute(0,2,1,3)
        ) 
        v = (
            self.v(v)
            .view(B,NK,self.num_heads,C // self.num_heads)
            .permute(0,2,1,3)
        ) 
        
        attn =(q @ k.transpose(-2,-1)) * self.scale
        attn =attn.softmax(dim=-1) 
        weights = attn
        attn = self.attn_drop(attn)
        # print(f'attn shape:{attn.shape}')
        # print(f'v shape :{v.shape}')
        x = (attn @ v).transpose(1,2).reshape(B,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x,weights
class SkipCrossAttention(nn.Module):
    def __init__(self,
        dim,
        num_heads,
        out_dim,
        dim_q=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,    
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.attn = CrossAttention(
            dim,
            out_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self,q,v):
        norm_q = self.norm1(q)
        q_1,_ = self.self_attn(norm_q)  
        q = q + self.drop_path(q_1) 

        norm_q = self.norm_q(q)
        norm_v = self.norm_v(v)

        q_2,weights =self.attn(norm_q,norm_v) 
        q = q + self.drop_path(q_2)

        q = q + self.drop_path(self.mlp(self.norm2(q)))

        return q,weights
class weight(nn.Module):
    def __init__(self,dim,num_points):
        super(weight, self).__init__()
        self.learnable_param = nn.Parameter(torch.empty(dim,num_points))
        nn.init.normal_(self.learnable_param, mean=0, std=0.01)

    def forward(self, x):
        return x * self.learnable_param

class AdaptivePointNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()
        Conv = nn.Conv1d

        self.norm = nn.InstanceNorm1d(in_channel)
        self.style = Conv(style_dim, in_channel * 2, 1)

        self.style.weight.data.normal_()
        self.style.bias.data.zero_()

        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = input
        out = gamma * out + beta

        return out
class BaseModel(nn.Module):
    def __init__(self,args):
            super(BaseModel, self).__init__()
            self.embed_dim = args['embed_dim']
            self.depth = args['depth']
            self.out_dim = args['out_dim']
            self.num_heads = args['num_heads']
            self.mlp_ratio = args['mlp_ratio']
            self.qkv_bias = args['qkv_bias']
            self.qk_scale = args['qk_scale']
            self.drop_rate = args['drop_rate']
            self.attn_drop_rate = args['attn_drop_rate']
            self.drop_path_rate = args['drop_path_rate']
            self.num_points = args['num_points']
            self.dec_dim = args['dec_dim']
            self.yaml = args['yaml']
     
            #encoder
            from models.encoder import resnet18
            self.encoder = resnet18(pretrained=False, num_classes=1000)
            self.linear = nn.Linear(1000, 768)
            #pc_encoder
            config_path = self.yaml
            config = munch.munchify(yaml.safe_load(open(config_path)))
            pc_encoder = builder.model_builder(config.model)
            pc_encoder.load_model_from_ckpt(config.ckpts)
            self.pc_encoder=pc_encoder
            #get_spherepoints
            # self.sphere_points = get_spherepoints(2048,0.5)
            self.num_points = self.num_points
            #attention
            self.blocks = nn.ModuleList(
                [
                    SkipCrossAttention(
                        dim=self.embed_dim,
                        out_dim=self.out_dim,
                        num_heads=self.num_heads,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=self.qkv_bias,
                        qk_scale=self.qk_scale,
                        drop=self.drop_rate,
                        attn_drop=self.attn_drop_rate,
                        drop_path_rate=self.drop_path_rate,
                    )
                    for _ in range(self.depth)
                ]
            )
            #decoder      
            dec_dim = self.dec_dim
            self.bn1 = nn.BatchNorm1d(dec_dim[1])
            self.bn2 = nn.BatchNorm1d(dec_dim[2])
            self.bn3 = nn.BatchNorm1d(dec_dim[3])
            self.bn4 = nn.BatchNorm1d(dec_dim[4])
            self.bn5 = nn.BatchNorm1d(dec_dim[5])
            
            self.conv1 = nn.Sequential(nn.Conv1d(dec_dim[0], dec_dim[1], kernel_size=1, bias=False),
                                           self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
            self.conv2 = nn.Sequential(
                                nn.Conv1d(dec_dim[1],dec_dim[2], kernel_size=1, bias=False),
                                self.bn2,
                                nn.LeakyReLU(negative_slope=0.2),
            )
            self.conv3 = nn.Sequential(
                                nn.Conv1d(dec_dim[2],dec_dim[3], kernel_size=1, bias=False),
                                self.bn3,
                                nn.LeakyReLU(negative_slope=0.2),
            )
            self.conv4 = nn.Sequential(
                                nn.Conv1d(dec_dim[3],dec_dim[4], kernel_size=1, bias=False),
                                self.bn4,
                                nn.LeakyReLU(negative_slope=0.2),
            )
            self.conv5 = nn.Sequential(
                                nn.Conv1d(dec_dim[4],dec_dim[5], kernel_size=1, bias=False),
                                self.bn5,
                                nn.LeakyReLU(negative_slope=0.2),
            )
                                   
            self.tail = nn.Sequential(
                                nn.Conv1d(dec_dim[5],3, kernel_size=1, bias=False),
                                nn.Tanh())
            self.map1 = weight(dec_dim[0],self.num_points)
            self.map2 = weight(dec_dim[0],self.num_points)
            self.map3 = weight(dec_dim[0],self.num_points)
            self.map4 = weight(dec_dim[0],self.num_points)
            self.map5 = weight(dec_dim[0],self.num_points)
            self.map6 = weight(dec_dim[0],self.num_points)

            self.adain1 = AdaptivePointNorm(dec_dim[1],dec_dim[0])
            self.adain2 = AdaptivePointNorm(dec_dim[2],dec_dim[0])
            self.adain3 = AdaptivePointNorm(dec_dim[3],dec_dim[0])
            self.adain4 = AdaptivePointNorm(dec_dim[4],dec_dim[0])
            self.adain5 = AdaptivePointNorm(dec_dim[5],dec_dim[0])
            self.adain6 = AdaptivePointNorm(3,dec_dim[0])
                                   
                
    def forward(self, images,xyz):
        B,_,_ = xyz.shape
        n = 24 
        attn_weights = []
        with torch.no_grad():
            pc_feat = self.pc_encoder(xyz.contiguous()).reshape(B,n,-1)
        Im_feat = self.linear(self.encoder(images)).reshape(B,n,-1)

        for i, blk in enumerate(self.blocks):
            if i % 2:  
                Im_feat, weights_i = blk(q=Im_feat, v=pc_feat)
            else:  
                pc_feat, weights_i = blk(q=pc_feat, v=Im_feat)
            attn_weights.append(weights_i)
        
        Im_feat = Im_feat.reshape(B,-1)
        Im_feat = Im_feat.unsqueeze(2).repeat(1,1,2048)
        style = Im_feat

        Im_feat = self.map1(Im_feat)  
        x = self.conv1(Im_feat) 
        x1 = self.adain1(x,Im_feat)

        Im_feat2 = self.map2(style) 
        x1 = self.conv2(x1)
        x2 = self.adain2(x1,Im_feat2)# 256

        Im_feat3 = self.map3(style)
        x2 = self.conv3(x2)
        x3 = self.adain3(x2,Im_feat3) #128

        Im_feat4 = self.map4(style)
        x3 = self.conv4(x3)
        x4 = self.adain4(x3,Im_feat4)

        Im_feat5 = self.map5(style)
        x4 = self.conv5(x4)
        x5 = self.adain5(x4,Im_feat5)

        Im_feat6 = self.map6(style) 
        outputs = self.tail(x5)
        outputs = self.adain6(outputs,Im_feat6).transpose(2,1)

        return outputs.transpose(2,1).contiguous()

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

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
def save_model_2(path, net, net_d=None):
    if net_d is not None:
        torch.save({'net_state_dict': net.module.state_dict(),
                    'D_state_dict': net_d.module.state_dict()}, path)
    else:
        torch.save({'net_state_dict': net.module.state_dict()}, path)
def save_model(path, net, net_d=None):
    if net_d is not None:
        torch.save({'net_state_dict': net.state_dict(),
                    'D_state_dict': net_d.state_dict()}, path)
    else:
        torch.save({'net_state_dict': net.state_dict()}, path)

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
if __name__ =='__main__':
    from dataset_svr.trainer_dataset import get_spherepoints
    from thop import profile
    q =torch.randn(1,3,224,224).to('cuda:0')
    p = torch.tensor(get_spherepoints(2048, 0.5)).unsqueeze(0).to('cuda:0')
    p = p.to(torch.float32)

    print('done')
    Model1 = BaseModel().to('cuda:0')
    # Model1.train()
    # out = Model1(q,p)
    # for param in Model1.pc_encoder.parameters():
    #     param.requires_grad = False

    # total_params = sum(p.numel() for p in Model1.parameters()  if p.requires_grad)
    # print("params: %.2f M" % (total_params / 1e6))


