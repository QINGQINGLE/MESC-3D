'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Le Xue
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ULIP_utils import all_gather_batch,get_rank

loss = torch.nn.KLDivLoss(reduction='batchmean')

def KL_loss(student_features,teacher_features):
    feature_loss = loss(F.log_softmax(student_features, dim=-1), F.softmax(teacher_features, dim=-1))
    return feature_loss
def contrastive_loss(e_s, e_t):
    """
    Compute the contrastive loss between shape (e_s) and image (e_t) embeddings.
    
    Args:
    e_s: Tensor of shape (N, C)
    e_t: Tensor of shape (N, C)
    
    Returns:
    loss: Contrastive loss value
    """
    N, C = e_s.size()
    
    # Normalize embeddings
    e_s = F.normalize(e_s, p=2, dim=1)
    e_t = F.normalize(e_t, p=2, dim=1)
    
    # Compute the similarity matrix
    sim_matrix = torch.matmul(e_s, e_t.T)  # Shape: (N, N)
    
    # Compute the log probabilities
    log_prob_s = F.log_softmax(sim_matrix, dim=1)  # Shape: (N, N)
    log_prob_t = F.log_softmax(sim_matrix.T, dim=1)  # Shape: (N, N)
    
    # Contrastive loss
    loss_s = -torch.mean(torch.diag(log_prob_s))
    loss_t = -torch.mean(torch.diag(log_prob_t))
    
    # Total loss
    loss = (loss_s + loss_t) / 2
    
    return loss

class ULIPWithImageLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, outputs):
        pc_embed = outputs['pc_embed']
        text_embed = outputs['text_embed']
        image_embed = outputs['image_embed']
        logit_scale = outputs['logit_scale']
        local_batch_size = pc_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * get_rank() + torch.arange(
                local_batch_size, device=pc_embed.device
            )
            self.last_local_batch_size = local_batch_size

        #normalized features
        pc_embed = F.normalize(pc_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        # gather features from all GPUs
        pc_embed_all, text_embed_all, image_embed_all = \
            all_gather_batch([pc_embed, text_embed, image_embed])
        
        #cosine similarity as logits #[batch_size,batch_size]

        logits_per_pc_text = logit_scale * pc_embed @ text_embed_all.t()
        # logits_per_image_text = logit_scale * image_embed @ text_embed_all.t()


        ulip_pc_text =  logit_scale - torch.diag(logits_per_pc_text).mean()
        # ulip_pc_image = logit_scale - torch.diag(logits_per_pc_image).mean()
        # ulip_text_image = logit_scale - torch.diag(logits_per_image_text).mean()
        # kl = KL_loss(text_embed,pc_embed).mean()
        con_loss = contrastive_loss(pc_embed,text_embed_all)
        # ctr = contrastive_loss(text_embed,pc_embed)
        # loss = ulip_pc_text + 0.2*ulip_text_image
        # loss = ulip_pc_text + 0.05 *ulip_text_image
        # loss = ulip_pc_text + kl
        loss = con_loss
        return {'loss': loss, 'ulip_loss': loss, 'ulip_text_pc_sim': ulip_pc_text, 'ulip_text_image_sim': 0}
