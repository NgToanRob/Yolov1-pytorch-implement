import torch
import torch.nn as nn
from utils import iOU

class YoloLoss(nn.Module):
    
    def __init__(self, split = 7, Bbox= 2, num_cls =20):
        
        super(YoloLoss, self).__init__()
        self.S = split
        self.B = Bbox
        self.C = num_cls
        self.mse = nn.MSELoss(reduction = "sum")
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
        
    def forward(self, pred, target):
        
        pred = pred.reshape(-1, self.S, self.S, self.C + self.B*5)
        
        iou1 = iOU(pred[...,21:25], target[...,21:25])
        iou2 = iOU(pred[...,26:30], target[...,21:25])
        
        ious = torch.cat([iou1.unsqueeze(0), iou2.unsqueeze(0)], dim = 0)#[[iou1],[iou2]]
        
        #bbox
        iou_max_val, best_bbox = torch.max(ious, dim = 0)
        
        #I_obj_ij
        actual_box = target[...,20].unsqueeze(3)#(-1,S,S,C+B*5) ==> (-1, S,S,1,C+B*5)
        
        
        #box coords
        box_pred = actual_box * (
            (
                best_bbox * pred[...,26:30]
                + (1 - best_bbox) *pred[...,21:25]
            )
        )
        
        box_pred[...,2:4] = torch.sign(box_pred[..., 2:4]) * (torch.sqrt(torch.abs(box_pred[...,2:4] + 1e-6)))
        
        box_target = actual_box * target[...,21:25]
        
        box_target[...,2:4] = torch.sqrt(box_target[..., 2:4])
        
        box_coord_loss = self.mse(
            
            torch.flatten(box_pred, end_dim=-2),
            torch.flatten(box_target, end_dim=-2)

        )
        
        #object loss
        pred_box = (best_bbox * pred[..., 25:26] + (1 - best_bbox) * pred[...,20:21])
        
        obj_loss = self.mse(
            torch.flatten(actual_box * pred_box),
            torch.flatten(actual_box * target[..., 20:21])
        )
        
        
        #no object loss
        no_obj_loss = self.mse(
            torch.flatten((1 - actual_box) * pred[..., 20:21],start_dim = 1),
            torch.flatten((1 - actual_box) * target[..., 20:21],start_dim = 1)
        )
        
        no_obj_loss += self.mse(
            torch.flatten((1 - actual_box) * pred[..., 25:26],start_dim = 1),
            torch.flatten((1 - actual_box) * target[..., 20:21],start_dim = 1)
        )
        
        #class loss
        class_loss = self.mse(
            torch.flatten((actual_box) * pred[..., :20],end_dim = -2),
            torch.flatten((actual_box) * target[..., :20],end_dim = -2)
        )
        
        loss = (
            
            self.lambda_coord * box_coord_loss+
            obj_loss+
            self.lambda_noobj * no_obj_loss+
            class_loss
        
        )
        
        return loss
