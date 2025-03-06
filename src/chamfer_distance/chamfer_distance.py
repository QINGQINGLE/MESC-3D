import torch
import importlib
from torch.utils.cpp_extension import load
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '5'
script_dir = os.path.dirname(__file__)

sources = [
    os.path.join(script_dir, "chamfer_distance.cpp"),
    os.path.join(script_dir, "chamfer_distance.cu"),
]
chamfer_found = importlib.find_loader("chamfer_3D") is not None
if not chamfer_found:
    ## Cool trick from https://github.com/chrdiller
    print("Jitting Chamfer 3D")
    from torch.utils.cpp_extension import load
    chamfer_3D = load(name="chamfer_3D",sources=sources)
    print("Loaded JIT 3D CUDA chamfer distance")
else:
    import chamfer_3D
    print("Loaded compiled 3D CUDA chamfer distance")


class ChamferDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
        # print("device:", device)
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n, dtype=torch.int)
        idx2 = torch.zeros(batchsize, m, dtype=torch.int)
        xyz1 = xyz1.to(device)
        xyz2 = xyz2.to(device)
        dist1 = dist1.to(device)
        dist2 = dist2.to(device)
        idx1 = idx1.to(device)
        idx2 = idx2.to(device)
        torch.cuda.set_device(device)
        chamfer_3D.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        # if not xyz1.is_cuda:
        #     chamfer_3D.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        # else:
        #     # dist1 = dist1.cuda()
        #     # dist2 = dist2.cuda()
        #     # idx1 = idx1.cuda()
        #     # idx2 = idx2.cuda()
        #     dist1 = dist1.to(device)
        #     dist2 = dist2.to(device)
        #     idx1 = idx1.to(device)
        #     idx2 = idx2.to(device)
        #     torch.cuda.set_device(device)
        #     # chamfer_3D.forward_cuda(xyz1, xyz2, dist1, dist2, idx1, sidx2)
        #     chamfer_3D.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)

        # ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2

    @staticmethod
    def backward(ctx, graddist1, graddist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors

        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()
        device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
        
        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())
        
        if not graddist1.is_cuda:
            chamfer_3D.backward(
                xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2
            )
        else:
            # gradxyz1 = gradxyz1.cuda()
            # gradxyz2 = gradxyz2.cuda()
            gradxyz1 = gradxyz1.to(device)
            gradxyz2 = gradxyz2.to(device)
            # chamfer_3D.backward_cuda(
            #     xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2
            # )
            chamfer_3D.backward(
                xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2
            )

        return gradxyz1, gradxyz2


class ChamferDistance(torch.nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()
    def forward(self, xyz1, xyz2):
        return ChamferDistanceFunction.apply(xyz1, xyz2)
