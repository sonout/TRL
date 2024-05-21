import torch
import torch.nn as nn

import copy
from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule


from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule


class MoCo_fusion3(nn.Module):
    def __init__(self, encoder_road1, encoder_road2, encoder_cell, 
                 road1_nemb, road2_nemb, cell_nemb, nout,
                queue_size, mmt = 0.999, temperature = 0.07):
        super(MoCo_fusion3, self).__init__()
        
        self.queue_size = queue_size
        self.mmt = mmt
        self.temperature = temperature

        self.encoder_road1 = encoder_road1
        self.encoder_road2 = encoder_road2
        self.encoder_cell = encoder_cell

        # TODO: Embedding dimensions are different for road1, road2 and cell
        self.projection_head_road1 = MoCoProjectionHead(road1_nemb, road1_nemb, nout)
        self.projection_head_road2 = MoCoProjectionHead(road2_nemb, road2_nemb, nout)
        self.projection_head_cell = MoCoProjectionHead(cell_nemb, cell_nemb, nout)

        self.encoder_cell_mmt = copy.deepcopy(self.encoder_cell)
        self.encoder_road_mmt1 = copy.deepcopy(self.encoder_road1)
        self.encoder_road_mmt2 = copy.deepcopy(self.encoder_road2)

        self.projection_head_cell_mmt = copy.deepcopy(self.projection_head_cell)
        self.projection_head_road_mmt1 = copy.deepcopy(self.projection_head_road1)
        self.projection_head_road_mmt2 = copy.deepcopy(self.projection_head_road2)
        deactivate_requires_grad(self.encoder_cell_mmt)
        deactivate_requires_grad(self.encoder_road_mmt1)
        deactivate_requires_grad(self.encoder_road_mmt2)
        deactivate_requires_grad(self.projection_head_cell_mmt)
        deactivate_requires_grad(self.projection_head_road_mmt1)
        deactivate_requires_grad(self.projection_head_road_mmt2)

        self.criterionMoCo_road1 = NTXentLoss(temperature=temperature, memory_bank_size=queue_size)
        self.criterionMoCo_road2 = NTXentLoss(temperature=temperature, memory_bank_size=queue_size)
        self.criterionMoCo_cell = NTXentLoss(temperature=temperature, memory_bank_size=queue_size) 
        self.criterionMoCo_inter1 = NTXentLoss(temperature=temperature, memory_bank_size=queue_size)
        self.criterionMoCo_inter2 = NTXentLoss(temperature=temperature, memory_bank_size=queue_size)

    def forward_normal_road1(self, x):
        query = self.encoder_road1(**x)
        p = self.projection_head_road1(query)
        return p
    
    def forward_momentum_road1(self, x):
        key = self.encoder_road_mmt1(**x)
        p = self.projection_head_road_mmt1(key).detach()
        return p
    

    def forward_normal_road2(self, x):
        query = self.encoder_road2(**x)
        p = self.projection_head_road2(query)
        return p
    
    def forward_momentum_road2(self, x):
        key = self.encoder_road_mmt2(**x)
        p = self.projection_head_road_mmt2(key).detach()
        return p
    
    
    def forward_normal_cell(self, x):
        query = self.encoder_cell(**x)#.flatten(start_dim=1)
        p = self.projection_head_cell(query)
        return p

    def forward_momentum_cell(self, x):
        key = self.encoder_cell_mmt(**x)#.flatten(start_dim=1)
        p = self.projection_head_cell_mmt(key).detach()
        return p
    
    

    def forward(self, kwargs_q_road1, kwargs_k_road1, kwargs_q_road2, kwargs_k_road2, kwargs_q_cell, kwargs_k_cell ):
        
        ## Update all momentum
        current_epoch = 1
        momentum = cosine_schedule(current_epoch, 10, self.mmt, 1)
        update_momentum(self.encoder_road1, self.encoder_road_mmt1, m=momentum)
        update_momentum(self.projection_head_road1, self.projection_head_road_mmt1, m=momentum)
        update_momentum(self.encoder_road2, self.encoder_road_mmt2, m=momentum)
        update_momentum(self.projection_head_road2, self.projection_head_road_mmt2, m=momentum)
        update_momentum(self.encoder_cell, self.encoder_cell_mmt, m=momentum)
        update_momentum(self.projection_head_cell, self.projection_head_cell_mmt, m=momentum)
        
        ## Forward road1, road2, cell ##
        ## inter loss ##
        q_p_road1 = self.forward_normal_road1(kwargs_q_road1)
        k_p_road1 = self.forward_momentum_road1(kwargs_k_road1)
        loss_road1 = self.criterionMoCo_road1(q_p_road1, k_p_road1)

        q_p_road2 = self.forward_normal_road2(kwargs_q_road2)
        k_p_road2 = self.forward_momentum_road2(kwargs_k_road2)
        loss_road2 = self.criterionMoCo_road2(q_p_road2, k_p_road2)

        q_p_cell = self.forward_normal_cell(kwargs_q_cell)
        k_p_cell = self.forward_momentum_cell(kwargs_k_cell)
        loss_cell = self.criterionMoCo_cell(q_p_cell, k_p_cell)
        
        ## intra loss ##
        # 1. between road1 and road2
        loss_inter1 = self.criterionMoCo_inter1(q_p_road1, k_p_road2)
        # 2. between road2 and cell
        loss_inter2 = self.criterionMoCo_inter2(q_p_road2, k_p_cell)

        # loss = loss_cell + loss_road + inter_loss * 2

        loss = loss_cell + loss_road1 + loss_road2 + loss_inter1 + loss_inter2
        return loss
    
    def encode(self, kwargs_x_road1, kwargs_x_road2, kwargs_x_cell):
        z_road1 = self.encoder_road1(**kwargs_x_road1)
        z_road2 = self.encoder_road2(**kwargs_x_road2)
        z_cell = self.encoder_cell(**kwargs_x_cell)
        z = torch.cat((z_road1, z_road2, z_cell), dim=1)
        return z
    

