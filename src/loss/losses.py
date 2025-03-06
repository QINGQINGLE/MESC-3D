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

from ULIP_utils import all_gather_batch,get_rank

loss = torch.nn.KLDivLoss(reduction='batchmean')

def KL_loss(student_features,teacher_features):
    feature_loss = loss(F.log_softmax(student_features, dim=-1), F.softmax(teacher_features, dim=-1))
    return feature_loss

labels = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
def contrastive_loss(student_features, teacher_features, margin=1.0):
    distances = (student_features - teacher_features).pow(2).sum(1)  # 欧几里得距离的平方
    loss = (1 - labels) * distances + labels * F.relu(margin - distances.sqrt()).pow(2)
    return loss.mean()

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

        # normalized features
        # pc_embed = F.normalize(pc_embed, dim=-1, p=2)
        # text_embed = F.normalize(text_embed, dim=-1, p=2)
        # image_embed = F.normalize(image_embed, dim=-1, p=2)
        # # gather features from all GPUs
        # pc_embed_all, text_embed_all, image_embed_all = \
        #     all_gather_batch([pc_embed, text_embed, image_embed])
        # cosine similarity as logits
        # logits_per_pc_text = logit_scale * pc_embed @ text_embed_all.t()
        # logits_per_text_pc = logit_scale * text_embed @ pc_embed_all.t()
        # logits_per_text_image = logit_scale * text_embed @ image_embed_all.t()
        # logits_per_image_text = logit_scale * image_embed @ text_embed_all.t()

        # logits_per_pc_text = F.cosine_similarity(text_embed,pc_embed)
        logits_per_text_image = F.cosine_similarity(text_embed,image_embed)
        # ulip_pc_text =  logit_scale - logits_per_pc_text.mean()
        ulip_text_image = logit_scale - logits_per_text_image.mean()

        # loss =  0.9*ulip_pc_text + 0.1*ulip_text_image
        loss = ulip_text_image

        return {'loss': loss, 'ulip_loss': loss, 'ulip_text_pc_sim':0, 'ulip_text_image_sim': ulip_text_image}
