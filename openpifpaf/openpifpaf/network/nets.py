import copy
import logging
import os.path
import time
from collections import OrderedDict

import torch

from openpifpaf.network.neck_transformer import JointTransformer, SoloTransformer

LOG = logging.getLogger(__name__)

MODEL_MIGRATION = set()
from torch.nn.functional import mse_loss


class Shell(torch.nn.Module):
    def __init__(self, base_net, head_nets, depth_net, *,
                 process_input=None, process_heads=None, mode='rgb', use_transformer=False, depth_checkpoint=None):
        """
        Args:
            base_net: backbone for rgb image
            head_nets: head nets that preds the
            depth_net:
            process_input:
            process_heads:
            mode: rgb,depth or combine
        """
        super().__init__()
        assert mode in ['rgb', 'depth', 'combine'], "mode must be rgb,depth or combine"
        self.mode = mode
        self.use_transformer = use_transformer

        if self.mode == 'depth' or self.mode=='combine':

            # if self.use_transformer:
            #     self.depth_transformer = JointTransformer(in_channels=256, out_channels=256, embed_dim=256, mlp_ratio=8.,
            #                                         num_heads=8, depth=6, drop_path_rate=0.0)
            self.depth_basenet = depth_net
            if self.mode=='depth':
                self.set_depth_head_nets(head_nets)
            else:
                self.set_depth_head_nets(copy.deepcopy(head_nets))
            self.process_heads = process_heads
            self.depth_to_rgb = torch.nn.Linear(1, 3)
        if self.mode == 'rgb'or self.mode=='combine':

            # if self.use_transformer:
            #     self.transformer = JointTransformer(in_channels=256, out_channels=256, embed_dim=256, mlp_ratio=8.,
            #                                         num_heads=8, depth=6, drop_path_rate=0.0)
            self.base_net = base_net
            self.process_input = process_input
            self.process_heads = process_heads
            self.set_head_nets(head_nets)
        # else:
        #     self.base_net = base_net
        #     self.depth_basenet = depth_net
        #     self.depth_to_rgb = torch.nn.Linear(1, 3)
        #     self.set_depth_head_nets(head_nets)
        #     self.set_head_nets(head_nets)
        #     self.process_heads = process_heads
        #     self.process_input = process_input
            #
            # if self.use_transformer:
            #     self.transformer = JointTransformer(in_channels=256, out_channels=256, embed_dim=256, mlp_ratio=8.,
            #                                         num_heads=8, depth=6, drop_path_rate=0.0)
            #     self.depth_transformer = JointTransformer(in_channels=256, out_channels=256, embed_dim=256, mlp_ratio=8.,
            #                                         num_heads=8, depth=6, drop_path_rate=0.0)

        # Super mega-ugly hack
        if self.mode == 'depth' or self.mode=='combine':
            self.depth_neck = self.get_neck(self.depth_basenet, self.depth_head_nets)
        if self.mode == 'rgb'or self.mode=='combine':
            self.neck = self.get_neck(self.base_net, self.head_nets)
        # else:
        #     self.depth_neck = self.get_neck(self.depth_basenet, self.depth_head_nets)
        #     self.neck = self.get_neck(self.base_net, self.head_nets)
        if self.mode == 'combine':
            assert depth_checkpoint is not None, f"depth_checkpoint {depth_checkpoint} must be provided when mode is combine"
            assert os.path.exists(depth_checkpoint), "depth_checkpoint must be provided when mode is combine"
            self.depth_checkpoint = depth_checkpoint
            depth_pretrained_weight = torch.load(self.depth_checkpoint)
            self.depth_basenet.load_state_dict(depth_pretrained_weight['model'].depth_basenet.state_dict())
            # self.depth_to_rgb.load_state_dict(depth_pretrained_weight['model'].depth_to_rgb.state_dict())
            self.depth_head_nets[0].load_state_dict(depth_pretrained_weight['model'].depth_head_nets[0].state_dict(),
                                                    strict=False)
            raf_head_dict={}
            for key in depth_pretrained_weight['model'].depth_head_nets[1].state_dict().keys():
                if 'raf' not in key:
                    raf_head_dict[key]=depth_pretrained_weight['model'].depth_head_nets[1].state_dict()[key]
            self.depth_head_nets[1].load_state_dict(OrderedDict(raf_head_dict), strict=False)

            self.depth_model_list = [self.depth_basenet, self.depth_head_nets[0],
                                     self.depth_head_nets[1], self.depth_neck]
            for model in self.depth_model_list:
                if model is not None:
                    for param in model.parameters():
                        param.requires_grad = False
                    model.eval()

    def get_neck(self, base_net, head_nets):
        if getattr(head_nets[1], 'joint_transformer', False):
            depth_neck = JointTransformer(in_channels=base_net.out_features,
                                          out_channels=256, embed_dim=256, mlp_ratio=8.,
                                          num_heads=8, depth=6, drop_path_rate=0.0)
        elif getattr(head_nets[1], 'solo_transformer', False):
            depth_neck = SoloTransformer(in_channels=base_net.out_features,
                                         out_channels=256, embed_dim=256, mlp_ratio=8.,
                                         num_heads=8, depth=6, drop_path_rate=0.0)
        else:
            depth_neck = None
        return depth_neck

    def get_head_results(self, *args):
        head_nets = args[0]
        x = args[1]
        args = args[2:]
        head_outputs = []
        offset_output = []
        if len(args) >= 3:
            head_mask = args[3]
            targets = args[2]
            for hn_idx, (hn, m) in enumerate(zip(head_nets, head_mask)):
                if m:
                    if hn.__class__.__name__ == "RafHead" and hn.meta.refine_hm:
                        if isinstance(x, list) and len(x) > 1:
                            # if self.use_concat and hn_idx==1:
                            head_output, offset = hn(x[hn_idx % 2], targets[hn_idx], extra=head_outputs[0])
                            offset_output.append(offset)
                            head_outputs.append(head_output)
                            # else:
                            #     head_output,offset=hn(x[hn_idx], targets[hn_idx])
                            #     offset_output.append(offset)
                            #     head_outputs.append(head_output)
                        else:
                            # if self.use_concat and hn_idx==1:
                            head_output, offset = hn(x[hn_idx % 2], targets[hn_idx], extra=head_outputs[0])
                            offset_output.append(offset)
                            head_outputs.append(head_output)
                            # else:
                            #     head_output,offset=hn(x, targets[hn_idx])
                            #     offset_output.append(offset)
                            #     head_outputs.append(head_output)
                    else:
                        if getattr(self, 'neck', False) and self.neck is not None:
                            # if self.use_concat and hn_idx==1:
                            head_output, offset = hn(x[hn_idx], extra=head_outputs[0])
                            head_outputs.append(head_output)
                            offset_output.append(offset)
                            # else:
                            #     head_output,offset=hn(x[hn_idx])
                            #     head_outputs.append(head_output)
                            #     offset_output.append(offset)
                        elif isinstance(x, list) and len(x) > 1:
                            # if self.use_concat and hn_idx == 1:
                            head_output, offset = hn(x[hn_idx % 2], extra=head_outputs[0])
                            head_outputs.append(head_output)
                            offset_output.append(offset)
                            # else:
                            #     head_output, offset = hn(x[hn_idx % 2])
                            #     head_outputs.append(head_output)
                            #     offset_output.append(offset)
                        else:
                            # if self.use_concat and hn_idx == 1:
                            head_output, offset = hn(x, extra=head_outputs[0])
                            head_outputs.append(head_output)
                            offset_output.append(offset)
                            # else:
                            #     head_output,offset=hn(x,targets)
                            #     head_outputs.append(head_output)
                            #     offset_output.append(offset)
                            # head_outputs.append(hn(x, targets))
                # else:
                #     head_outputs.append(None)
            # head_outputs.extend([gt_depth])
            head_outputs = tuple(head_outputs)
        else:
            for hn_idx, hn in enumerate(head_nets):
                # print(hn.__class__)
                if getattr(self, 'neck', False) and self.neck is not None:
                    if isinstance(x, list) and len(x) > 1:
                        head_output, offset = hn(x[hn_idx % 2])
                        head_outputs.append(head_outputs)
                        offset_output.append(offset)
                    else:
                        head_output, offset = hn(x[hn_idx])
                        head_outputs.append(head_output)
                        offset_output.append(offset)
                        # head_outputs.append(hn(x[hn_idx]))
                    # head_outputs.append(hn(x[hn_idx]))
                elif isinstance(x, list) and len(x) > 1:
                    head_output, offset = hn(x[hn_idx % 2])
                    head_outputs.append(head_output)
                    offset_output.append(offset)
                else:
                    # hn_time_cost=time.time()
                    if hn_idx == 1:
                        head_output, offset = hn(x, extra=head_outputs[0])
                    else:
                        head_output, offset = hn(x)
                    # print("head {} time cost:{}".format(hn_idx,time.time()-hn_time_cost))
                    head_outputs.append(head_output)
                    offset_output.append(offset)
            # head_outputs.extend([None,None])
        if self.process_heads is not None:
            head_outputs = self.process_heads(head_outputs)
        return head_outputs, offset_output

    def set_head_nets(self, head_nets):

        if not isinstance(head_nets, torch.nn.ModuleList):
            head_nets = torch.nn.ModuleList(head_nets)

        for hn_i, hn in enumerate(head_nets):
            hn.meta.head_index = hn_i
            hn.meta.base_stride = self.base_net.stride

        self.head_nets = head_nets

    def set_depth_head_nets(self, head_nets):
        if not isinstance(head_nets, torch.nn.ModuleList):
            head_nets = torch.nn.ModuleList(head_nets)

        for hn_i, hn in enumerate(head_nets):
            hn.meta.head_index = hn_i
            hn.meta.base_stride = self.depth_basenet.stride
        self.depth_head_nets = head_nets

    def forward(self, *args):
        image_batch = args[0]
        # depth_image=args[-1]

        # if self.training:
        depth = args[1]

        if self.mode == 'depth' or self.mode == 'combine':
            if self.mode == 'combine' and self.training:
                with torch.no_grad():
                    # self.depth_to_rgb.eval()
                    depth_x = depth
                    # depth=depth.unsqueeze(-1)
                    # depth_x=depth.repeat(1,1,1,3).permute(0,3,1,2)
                    # depth_x=image_batch
                    # self.depth_basenet.eval()
                    depth_x = self.depth_basenet(depth_x)

                    if self.depth_neck is not None:
                        # self.depth_neck.eval()
                        depth_x = self.depth_neck(depth_x)
                    # self.depth_head_nets[0].eval()
                    # self.depth_head_nets[1].eval()
                    depth_head_outputs, depth_offsets = self.get_head_results(self.depth_head_nets, depth_x, args)
            elif self.training:
                # depth_x = self.depth_to_rgb(depth.unsqueeze(-1)).permute(0, 3, 1, 2)
                depth_x = depth
                depth_x = self.depth_basenet(depth_x)
                # print("depth resnet time:", time.time() - resnet_start )
                if self.depth_neck is not None:
                    depth_x = self.depth_neck(depth_x)
                # head_start=time.time()
                depth_head_outputs, depth_offsets = self.get_head_results(self.depth_head_nets, depth_x, args)
                # print('depth head time:',time.time()-head_start)
            else:
                depth_head_outputs = None
                depth_offsets=None
        if self.mode == 'rgb' or self.mode == 'combine':
            if self.process_input is not None:
                image_batch = self.process_input(image_batch)
            x = self.base_net(image_batch)
            if self.neck is not None:
                x = self.neck(x)
            head_outputs, offset_outputs = self.get_head_results(self.head_nets, x, args)
        if self.mode == 'depth':
            return depth_head_outputs, None
        elif self.mode == 'rgb':
            return head_outputs, None
        else:
            if self.training:
                assert depth_offsets is not None, 'depth_offsets is None'
                assert offset_outputs is not None, 'offset_outputs is None'
            if self.training:
                offset_mask=[torch.where(torch.abs(depth_offset)<10,1,0) for depth_offset in depth_offsets]
                # offset_mask=[torch.ones(depth_offset.shape)for depth_offset in depth_offsets[:1]]
                offset_loss = [mse_loss(depth_offsets[i].detach(), offset_outputs[i],reduction='none').mul(offset_mask[i]) for i in range(len(offset_outputs))]
                offset_loss = [torch.sum(offset_loss[i])/(torch.sum(offset_mask[i])+1) for i in range(len(offset_loss))]
                offset_loss=torch.mean(torch.stack(offset_loss))
            else:
                offset_loss=None
            # print(offset_loss)
            return head_outputs, offset_loss

        # head_outputs = []
        # if len(args) >= 3:
        #     head_mask = args[3]
        #     targets = args[2]
        #     for hn_idx, (hn, m) in enumerate(zip(head_nets, head_mask)):
        #         if m:
        #             if hn.__class__.__name__ == "RafHead" and hn.meta.refine_hm:
        #
        #                 if isinstance(x, list) and len(x) > 1:
        #                     head_outputs.append(hn((x[hn_idx % 2], targets[hn_idx])))
        #                 else:
        #                     head_outputs.append(hn((x, targets)))
        #             else:
        #                 if getattr(self, 'neck', False) and self.neck is not None:
        #                     head_outputs.append(hn(x[hn_idx]))
        #                 elif isinstance(x, list) and len(x) > 1:
        #                     head_outputs.append(hn(x[hn_idx % 2]))
        #                 else:
        #                     head_outputs.append(hn(x, targets))
        #         # else:
        #         #     head_outputs.append(None)
        #     # head_outputs.extend([gt_depth])
        #     head_outputs = tuple(head_outputs)
        # else:
        #     for hn_idx, hn in enumerate(self.head_nets):
        #         print(hn.__class__)
        #         if getattr(self, 'neck', False) and self.neck is not None:
        #             head_outputs.append(hn(x[hn_idx]))
        #         elif isinstance(x, list) and len(x) > 1:
        #             head_outputs.append(hn(x[hn_idx % 2]))
        #         else:
        #             head_outputs.append(hn(x))
        #     # head_outputs.extend([None,None])
        # if self.process_heads is not None:
        #     head_outputs = self.process_heads(head_outputs)

        # if has_combined and self.base_net.training:
        #     return head_outputs, combined_hm_preds
        # return head_outputs


class CrossTalk(torch.nn.Module):
    def __init__(self, strength=0.2):
        super().__init__()
        self.strength = strength

    def forward(self, *args):
        image_batch = args[0]
        if self.training and self.strength:
            rolled_images = torch.cat((image_batch[-1:], image_batch[:-1]))
            image_batch += rolled_images * self.cross_talk
        return image_batch


# pylint: disable=protected-access
def model_migration(net_cpu):
    model_defaults(net_cpu)

    if not hasattr(net_cpu, 'process_heads'):
        net_cpu.process_heads = None

    for m in net_cpu.modules():
        if not hasattr(m, '_non_persistent_buffers_set'):
            m._non_persistent_buffers_set = set()

    if not hasattr(net_cpu, 'head_nets') and hasattr(net_cpu, '_head_nets'):
        net_cpu.head_nets = net_cpu._head_nets
    if hasattr(net_cpu, 'head_nets'):
        for hn_i, hn in enumerate(net_cpu.head_nets):
            if not hn.meta.base_stride:
                hn.meta.base_stride = net_cpu.base_net.stride
            if hn.meta.head_index is None:
                hn.meta.head_index = hn_i
            if hn.meta.name == 'cif' and 'score_weights' not in vars(hn.meta):
                hn.meta.score_weights = [3.0] * 3 + [1.0] * (hn.meta.n_fields - 3)
    if hasattr(net_cpu, 'depth_head_nets'):
        for hn_i, hn in enumerate(net_cpu.depth_head_nets):
            if not hn.meta.base_stride:
                hn.meta.base_stride = net_cpu.base_net.stride
            if hn.meta.head_index is None:
                hn.meta.head_index = hn_i
            if hn.meta.name == 'cif' and 'score_weights' not in vars(hn.meta):
                hn.meta.score_weights = [3.0] * 3 + [1.0] * (hn.meta.n_fields - 3)
    for mm in MODEL_MIGRATION:
        mm(net_cpu)


def model_defaults(net_cpu):
    return
    import pdb;
    pdb.set_trace()
    for m in net_cpu.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            # avoid numerical instabilities
            # (only seen sometimes when training with GPU)
            # Variances in pretrained models can be as low as 1e-17.
            # m.running_var.clamp_(min=1e-8)
            # m.eps = 1e-3  # tf default is 0.001
            # m.eps = 1e-5  # pytorch default

            # This epsilon only appears inside a sqrt in the denominator,
            # i.e. the effective epsilon for division is much bigger than the
            # given eps.
            # See equation here:
            # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
            m.eps = 1e-4

            # smaller step size for running std and mean update
            m.momentum = 0.01  # tf default is 0.99
            # m.momentum = 0.1  # pytorch default

        elif isinstance(m, (torch.nn.GroupNorm, torch.nn.LayerNorm)):
            m.eps = 1e-4

        elif isinstance(m, (torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d)):
            m.eps = 1e-4
            m.momentum = 0.01
