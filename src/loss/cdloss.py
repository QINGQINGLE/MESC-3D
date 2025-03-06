import torch
import torch.nn as nn
from chamfer_distance.chamfer_distance import ChamferDistance
def fscore(dist1, dist2, threshold=0.01):
    """
        Calculates the F-score between two point clouds with the corresponding threshold value.
        :param dist1: Batch, N-Points
        :param dist2: Batch, N-Points
        :param th: float
        :return: fscore, precision, recall
    """
    # NB : In this depo, dist1 and dist2 are squared pointcloud euclidean distances, so you should adapt the threshold accordingly.
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2
class SimplificationLoss(nn.Module):
    def __init__(self):
        super(SimplificationLoss, self).__init__()

    def forward(self, ref_pc, samp_pc,calc_f1=False):

        cost_p1_p2, cost_p2_p1 = ChamferDistance()(samp_pc, ref_pc)
        cd_p = (torch.sqrt(cost_p1_p2).mean(1) + torch.sqrt(cost_p2_p1).mean(1)) / 2 # l1
        cd_t = (cost_p1_p2.mean(1) + cost_p2_p1.mean(1)) #l2
        if calc_f1:
            f1,_,_ =fscore(cost_p1_p2,cost_p2_p1)
            return cd_p,cd_t,f1

        else:
            return cd_p,cd_t