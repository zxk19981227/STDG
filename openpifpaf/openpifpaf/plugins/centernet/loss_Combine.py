import argparse
import logging
from torch.nn import MSELoss
import torch

from . import heads
from .losses_util import RegL1Loss_5D, RegLoss, NormRegL1Loss, RegWeightedL1Loss, FocalLoss, GIOULoss, RegL1Loss

LOG = logging.getLogger(__name__)
class Combine_Loss(torch.nn.Module):
    prescale = 1.0
    crit_reg = None
    crit_wh = None

    def __init__(self, head_net: heads.Objection_Relation_Combined_Head):
        super(Combine_Loss, self).__init__()
        self.n_vectors = head_net.meta.n_vectors
        self.n_scales = head_net.meta.n_scales
        self.giou_loss = head_net.meta.giou_loss
        self.crit = FocalLoss()
        self.crit_giou=GIOULoss()
        # self.depth=MSELoss()
        self.previous_losses = None
        LOG.debug('%s: n_vectors = %d, n_scales = %d',
                  head_net.meta.name, self.n_vectors, self.n_scales)

        self.field_names = (
            ['{}.{}.c'.format(head_net.meta.dataset, head_net.meta.name)]
            + ['{}.{}.vec{}'.format(head_net.meta.dataset, head_net.meta.name, i + 1)
               for i in range(self.n_vectors)]
            + ['{}.{}.scales{}'.format(head_net.meta.dataset, head_net.meta.name, i + 1)
               for i in range(self.n_scales)]
        )

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('CifDet_CN Loss')
        group.add_argument('--cifdet-cn-loss-prescale', default=cls.prescale, type=float)
        group.add_argument('--cifdet-cn-regression-loss', default='l1',
                           choices=['smoothl1', 'l1'],
                           help='type of regression loss')
        group.add_argument('--cifdet-cn-wh-loss', default='same',
                          choices=['cat_spec', 'dense', 'norm', 'same'],
                          help='type of wh loss')
        group = parser.add_argument_group('CenterNet Loss')
        group.add_argument('--cn-loss-prescale', default=cls.prescale, type=float)
        group.add_argument('--cn-regression-loss', default='l1',
                           choices=['smoothl1', 'l1'],
                           help='type of regression loss')
        group.add_argument('--cn-wh-loss', default='same',
                           choices=['cat_spec', 'dense', 'norm', 'same'],
                           help='type of wh loss')
        group.add_argument('--cn-ciou-loss', default=False, action='store_true',
                           help='use ciou loss')
    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.prescale = args.cifdet_cn_loss_prescale

        if args.cifdet_cn_regression_loss == 'smoothl1':
            cls.crit_reg = RegLoss()
        elif args.cifdet_cn_regression_loss == 'l1':
            cls.crit_reg = RegL1Loss_5D()
        elif args.cifdet_cn_regression_loss is None:
            cls.crit_reg = None
        else:
            raise Exception('unknown regression loss type {}'.format(args.cn_regression_loss))

        if args.cifdet_cn_wh_loss == 'same':
            cls.crit_wh = cls.crit_reg
        elif args.cifdet_cn_wh_loss == 'norm':
            cls.crit_wh = NormRegL1Loss()
        elif args.cifdet_cn_wh_loss == 'dense':
            cls.crit_wh = torch.nn.L1Loss(reduction='sum')
        elif args.cifdet_cn_wh_loss == 'cat_spec':
            cls.crit_wh = RegWeightedL1Loss()
        else:
            raise Exception('unknown wh loss type {}'.format(args.cn_wh_loss))
        cls.prescale = args.loss_prescale

        if args.cn_regression_loss == 'smoothl1':
            cls.crit_reg = RegLoss()
        elif args.cn_regression_loss == 'l1':
            cls.crit_reg = RegL1Loss()
        elif args.cn_regression_loss is None:
            cls.crit_reg = None
        else:
            raise Exception('unknown regression loss type {}'.format(args.cn_regression_loss))

        if args.cn_wh_loss == 'same':
            cls.crit_wh = cls.crit_reg
        elif args.cn_wh_loss == 'norm':
            cls.crit_wh = NormRegL1Loss()
        elif args.cn_wh_loss == 'dense':
            cls.crit_wh = torch.nn.L1Loss(reduction='sum')
        elif args.cn_wh_loss == 'cat_spec':
            cls.crit_wh = RegWeightedL1Loss()
        else:
            raise Exception('unknown wh loss type {}'.format(args.cn_wh_loss))
        cls.ciou = args.cn_ciou_loss

    def forward(self,outputs,targets):
        center_heads,raf_heads=outputs[:,:7],outputs[:,7:]
        return [self.centernet_forward(center_heads,targets[0]),self.raf_forward(raf_heads,targets[1])]

    def raf_forward(self,outputs,targets):
        output_hm = outputs[:, :-4]
        output_reg = outputs[:, -4:-2]
        output_wh = outputs[:, -2:]
        # print(f"output hm has nan is {torch.any(torch.isfinite(output_hm))}")
        # print(f"output reg has nan is {torch.any(torch.isfinite(output_reg))}")
        # print(f"output wh has nan is {torch.any(torch.isfinite(output_wh))}")
        target_hm = targets[0]
        target_reg = targets[1]
        target_wh = targets[2]
        target_mask = targets[3]
        target_ind = targets[4]

        hm_loss = self.crit(output_hm, target_hm) / 2.0

        wh_loss = self.crit_reg(
            output_wh, target_mask,
            target_ind, target_wh) / 2.0

        off_loss = self.crit_reg(output_reg, target_mask,
                                 target_ind, target_reg) / 2.0

        all_losses = [hm_loss] + [wh_loss] + [off_loss]
        if self.giou_loss or self.ciou:
            giou_loss = self.crit_giou(
                output_wh, output_reg, target_mask,
                target_ind, target_wh, target_reg) / 2.0
            all_losses += [giou_loss]

        if not all(torch.isfinite(l).item() if l is not None else True for l in all_losses):
            raise Exception('found a loss that is not finite: {}, prev: {}'
                            ''.format(all_losses, self.previous_losses))
        self.previous_losses = [float(l.item()) if l is not None else None for l in all_losses]

        return all_losses
    def centernet_forward(self, outputs, targets):
        hm_loss, wh_loss, off_loss = 0, 0, 0
        # outputs,depth_information=outputs
        # print(f"targets is {targets.shape}")
        # targets,depth_gt=targets
        # targets=targets

        output_hm = outputs[:, :, 0:1].sigmoid_()+1e-10
        output_reg = outputs[:, :, 1:3]
        output_wh = outputs[:, :, 3:5]

        if self.n_scales>0:
            output_scale1 = torch.nn.functional.softplus(outputs[:, :, 5:6])
            output_scale2 =  torch.nn.functional.softplus(outputs[:, :, 6:7])
        target_hm = targets[:, :, 0:1]
        target_reg = targets[:, :, 1:3]
        target_reg = target_reg.permute(0, 1, 3, 4, 2).contiguous().view(target_reg.shape[0], target_reg.shape[1]*target_reg.shape[3]*target_reg.shape[4],-1)
        target_wh = targets[:, :, 3:5]
        target_wh = target_wh.permute(0, 1, 3, 4, 2).contiguous().view(target_wh.shape[0], target_wh.shape[1]*target_wh.shape[3]*target_wh.shape[4],-1)
        target_mask = (target_wh !=0).sum(2)//2
        target_ind = torch.arange(target_mask.shape[1]).repeat(target_mask.shape[0],1).to(target_hm.device)
        target_ind[target_mask == 0] = 0
        if self.n_scales>0:
            target_scale1 = targets[:, :, 7:8]
            target_scale1 = target_scale1.permute(0, 1, 3, 4, 2).contiguous().view(target_scale1.shape[0], target_scale1.shape[1]*target_scale1.shape[3]*target_scale1.shape[4],-1)
            target_scale2 = targets[:, :, 8:9]
            target_scale2 = target_scale2.permute(0, 1, 3, 4, 2).contiguous().view(target_scale2.shape[0], target_scale2.shape[1]*target_scale2.shape[3]*target_scale2.shape[4],-1)

            scale_loss1 = self.crit_reg(
                output_scale1, target_mask,
                target_ind, target_scale1)/2.0
            scale_loss2 = self.crit_reg(
                output_scale2, target_mask,
                target_ind, target_scale2)/2.0
        # print(f"output hm is nan is {torch.any(torch.isnan(output_hm))}")
        # print(f"output hm is negative is {torch.any(output_hm<0)}")
        # print(f"target hm is nan is {torch.any(torch.isnan(target_hm))}")
        hm_loss = self.crit(output_hm, target_hm)/2.0

        # target_ind = torch.zeros_like(target_mask)
        # for i in range(target_ind.shape[0]):
        #     target_ind[i] = target_indices[1][target_indices[0]==i]

        wh_loss = self.crit_wh(
            output_wh, target_mask,
            target_ind, target_wh)/2.0

        off_loss = self.crit_reg(output_reg, target_mask,
                             target_ind, target_reg)/2.0

        all_losses = [hm_loss] + [wh_loss] + [off_loss]#+[depth_loss]
        # print(f"depth loss is {depth_loss}")
        if self.n_scales>0:
            all_losses = [hm_loss] + [wh_loss] + [off_loss] + [scale_loss1] + [scale_loss2]#+[depth_loss]
        if self.giou_loss:
            import pdb; pdb.set_trace()
        if not all(torch.isfinite(l).item() if l is not None else True for l in all_losses):
            raise Exception('found a loss that is not finite: {}, prev: {}'
                            ''.format(all_losses, self.previous_losses))
        self.previous_losses = [float(l.item()) if l is not None else None for l in all_losses]

        return all_losses