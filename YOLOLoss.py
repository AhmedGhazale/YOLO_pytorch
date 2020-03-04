import torch
import torch.nn.functional as F
import numpy as np
from YoloLoss import yoloLoss
class YOLOLoss(torch.nn.Module):

    def __init__(self, s, b, l_coord = 5, l_noobj = .5):
        super(YOLOLoss, self).__init__()
        self.s = s
        self.b = b
        self.l_coord = l_coord
        self.l_noobj  = l_noobj

    def compute_iou(self,boxes1, boxes2):
        #
        # boxes1_xy = torch.stack([boxes1[...,0] /self.s - .5 * boxes1[...,2],
        #                       boxes1[...,1] /self.s - .5 * boxes1[...,3],
        #                       boxes1[...,0] /self.s + .5 * boxes1[...,2],
        #                       boxes1[...,1] /self.s + .5 * boxes1[...,3]], dim=-1)
        #
        # boxes2_xy = torch.stack([boxes2[..., 0] / self.s - .5 * boxes2[..., 2],
        #                       boxes2[..., 1] / self.s - .5 * boxes2[..., 3],
        #                       boxes2[..., 0] / self.s + .5 * boxes2[..., 2],
        #                       boxes2[..., 1] / self.s + .5 * boxes2[..., 3]], dim=-1)
        boxes1_xy = torch.zeros_like(boxes1)
        boxes1_xy[...,:2] = boxes1[...,:2] / 14. - .5 * boxes1[...,2:4]
        boxes1_xy[...,2:4] = boxes1[...,:2] / 14. + .5 * boxes1[...,2:4]

        boxes2_xy = torch.zeros_like(boxes2)
        boxes2_xy[..., :2] = boxes2[..., :2] / 14. - .5 * boxes2[..., 2:4]
        boxes2_xy[..., 2:4] = boxes2[..., :2] / 14. + .5 * boxes2[..., 2:4]

        tl=torch.max(boxes1_xy[...,:2],boxes2_xy[...,:2])
        br=torch.min(boxes1_xy[...,2:],boxes2_xy[...,2:])

        intersection=torch.clamp(br-tl,0)

        intersection_area=intersection[...,0] * intersection[...,1]

        boxes1_area=boxes1[...,2] * boxes1[...,3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        #union_area=torch.clamp(boxes1_area + boxes2_area - intersection_area,min = 1e-9 )
        union_area = boxes1_area + boxes2_area - intersection_area

        #print((intersection_area/ union_area).shape)
        return  intersection_area/ union_area

    def forward(self, pred, true):

        batch_size = pred.shape[0]
        pred_boxes = pred[...,:5*self.b].view(-1,self.s,self.s,self.b,5).contiguous()
        pred_coord = pred_boxes[...,0:4]
        pred_conf = pred_boxes[...,4]
        pred_classes = pred[...,5*self.b:]

        true_boxes = true[..., :5 * self.b].view(-1, self.s, self.s, self.b, 5).contiguous()
        true_coord = true_boxes[..., 0:4]
        true_conf = true_boxes[..., 4]
        true_classes = true[..., 5 * self.b:]

        # class loss
        obj_mask = true_conf[...,0]
        no_obj_mask = torch.ones_like(obj_mask) - obj_mask
        class_loss = torch.pow(pred_classes- true_classes,2).sum(-1) * obj_mask

        # coord_loss

        ious = self.compute_iou(pred_coord, true_coord)

        val, idx = ious.max(-1,keepdim=True)


        #iou_mask = ious==val
        #eq = iou_mask[...,0] == iou_mask[...,1]

        # iou_mask = torch.zeros_like(ious)
        # for i in range(16):
        #     for j in range(14):
        #         for k in range(14):
        #             iou_mask[i,j,k,idx[i,j,k]] = 1

        iou_mask = torch.zeros_like(ious).scatter_(-1,idx,1)
        
       
        #mm = ious[...,0 ] > ious[...,1]
        #import ipdb;ipdb.set_trace()

        coord_obj_mask = iou_mask * true_conf
        coord_noobj_mask = (torch.ones_like(iou_mask) - iou_mask) * true_conf

        #coord_loss = coord_obj_mask * (F.mse_loss(pred_coord[...,0:2], true_coord[...,0:2], reduction='none') + F.mse_loss(torch.sqrt(pred_coord[...,2:4]), torch.sqrt(true_coord[...,2:4]), reduction='none') ).sum(-1)

        coord_loss = coord_obj_mask * torch.pow(pred_coord[...,0:2]- true_coord[...,0:2],2).sum(-1) + coord_obj_mask * torch.pow(torch.sqrt(pred_coord[...,2:4])- torch.sqrt(true_coord[...,2:4]),2).sum(-1)
        # conf loss
        no_obj_conf_loss =  no_obj_mask * torch.pow(pred_conf,2).sum(-1)
        obj_conf_loss =  (coord_obj_mask * F.mse_loss(pred_conf, ious,reduction='none')).sum(-1)
        no_obj_conf_loss_2 = (coord_noobj_mask * torch.pow(pred_conf,2)).sum(-1)

        return (torch.sum(class_loss) + torch.sum(coord_loss) * self.l_coord + 2 * torch.sum(obj_conf_loss) + torch.sum(no_obj_conf_loss_2) + torch.sum(no_obj_conf_loss) * self.l_noobj)/batch_size, torch.sum(coord_loss)



    






