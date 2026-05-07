# ------------------------------------------------------------------------------
# Class all model definitions should descend from.
# ------------------------------------------------------------------------------


import torch.nn as nn
import torch.optim as optim
from mp.models.segmentation.segformer import SegFormer
from mp.models.segmentation.vision_transformer import SwinUnet
from mp.models.segmentation.unet_fepegar import UNet2D


class Model(nn.Module):

    def __init__(self, input_shape=(1, 32, 32), output_shape=2):
        super(Model, self).__init__()
        self.backbone_old = None
        self.backbone_new = None
        self.input_shape = input_shape
        self.output_shape = output_shape

    def init_backbone(self):
        # print("backbone is segformer")
        self.backbone_new = SegFormer(num_classes=2)
        # self.backbone_new = SwinUnet(input_shape=192)
        # self.backbone_new = UNet2D(
        #     self.input_shape,
        #     2,
        #     dropout=0,
        #     monte_carlo_dropout=0,
        #     preactivation=False,
        # )
        self.backbone_old = None

    def preprocess_input(self, x):
        r"""E.g. pretrained features. Override if needed."""
        return x

    def freeze_encoder(self):
        for param in self.backbone_new.mit.parameters():
            param.requires_grad = False

    def freeze_new_decoder(self):
        for param in self.backbone_new.decoder.parameters():
            param.requires_grad = False

    def unfreeze_new_decoder(self):
        for param in self.backbone_new.decoder.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone_new(x)

    def forward_old(self, x):

        return self.backbone_old(x)

    def freeze_backbone(self, backbone):

        for param in backbone.parameters():
            param.requires_grad = False
        return backbone

    def finish(self):
        r"""Finish training, store current backbone as old backbone"""
        backbone_new_state_dict = self.backbone_new.state_dict()
        device = next(self.backbone_new.parameters()).device

        self.backbone_old = SegFormer()
        self.backbone_old.load_state_dict(backbone_new_state_dict)
        self.backbone_old = self.freeze_backbone(self.backbone_old)

        self.backbone_old.to(device)

        self.reset_bn_stats_new_model()

    def reset_bn_stats_new_model(self):
        for module in self.backbone_new.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                module.running_mean.zero_()
                module.running_var.fill_(1)

    def set_optimizers(self, optimizer=optim.SGD, lr=1e-4, weight_decay=1e-4):
        if optimizer == optim.SGD:
            self.backbone_optim = optimizer(self.backbone_new.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.backbone_optim = optimizer(self.backbone_new.parameters(), lr=lr)
