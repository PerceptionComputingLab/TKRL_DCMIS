from mp.models.model import Model
import torch.optim as optim
import torch
from copy import deepcopy
import torch.nn.functional as F
from mp.eval.inference.predict import softmax
import torch.nn as nn
from mp.models.segmentation.segformer_head import SegFormerHead


class TKRL(Model):

    def __init__(
        self,
        input_shape=(1, 256, 256),
        nr_labels=2,
    ):
        r"""Constructor

        Args:
            input_shape (tuple of int): input shape of the images
            nr_labels (int): number of labels for the segmentation
            unet_dropout (float): dropout probability for the U-Net
            unet_monte_carlo_dropout (float): monte carlo dropout probability for the U-Net
            unet_preactivation (boolean): whether to use U-Net pre-activations
        """
        super(TKRL, self).__init__()

        self.input_shape = input_shape
        self.nr_labels = nr_labels
        # Initialize mask tokens for various head dimensions
        self.mask_token = nn.Parameter(torch.zeros(1))
        self.mae_decoder = SegFormerHead(num_classes=1, in_channels=[32, 64, 160, 256], embedding_dim=192)
        self.probes_mask = None

        self.init_backbone()

    def set_optimizers(self, optimizer=optim.SGD, lr=1e-4, weight_decay=1e-4):
        if optimizer == optim.SGD:
            self.backbone_optim = optimizer(
                [
                    {"params": self.backbone_new.parameters(), "lr": lr},
                    {"params": self.mask_token, "lr": lr},
                    {"params": self.mae_decoder.parameters(), "lr": lr},
                ],
                weight_decay=weight_decay,
            )
        else:
            self.backbone_optim = optimizer(
                [
                    {"params": self.backbone_new.parameters(), "lr": lr},
                    {"params": self.mask_token, "lr": lr},
                    {"params": self.mae_decoder.parameters(), "lr": lr},
                ]
            )

    def get_features_variance_ranks(self, x, mask_ratio=[0.10, 0.10, 0.10, 0.10], random=False):

        img_size = x.shape[2]

        encoder_new = deepcopy(self.backbone_new.mit)
        encoder_old = deepcopy(self.backbone_old.mit)
        # detach
        for param in encoder_new.parameters():
            param.detach_()
        for param in encoder_old.parameters():
            param.detach_()
        features_new = encoder_new(x)
        features_old = encoder_old(x)
        mask_chosed_list = []
        for i in range(len(features_new)):
            B, C, H, W = features_new[i].shape
            if random:
                # ========== 随机 mask 模式 ==========
                num_mask = int(H * W * mask_ratio[i])
                rand_vals = torch.rand(B, H * W, device=features_new[i].device)
                _, rand_indices = torch.sort(rand_vals, descending=True)
                masked_indices = rand_indices[:, :num_mask]
                mask_chosed = torch.zeros((B, H * W), device=features_new[i].device)
                mask_chosed.scatter_(1, masked_indices, 1)
                mask_chosed = mask_chosed.view(B, H, W)
            else:
                feature_new = features_new[i].flatten(2).transpose(1, 2)
                feature_old = features_old[i].flatten(2).transpose(1, 2)
                variance = torch.var(feature_new - feature_old, dim=2)
                # variance = torch.mean(variance, dim=2)
                sorted_variance, sorted_indices = torch.sort(variance, descending=True, dim=1)
                top_k = int(sorted_indices.shape[1] * mask_ratio[i])
                masked_indices = sorted_indices[:, :top_k]
                mask_chosed = torch.zeros_like(sorted_variance)
                mask_chosed.scatter_(1, masked_indices, 1)
                mask_chosed = mask_chosed.view(B, H, W)

            up_mask = F.interpolate(mask_chosed.unsqueeze(1), size=(img_size, img_size), mode="nearest")
            mask_chosed_list.append(up_mask)

        probes_mask = torch.max(torch.stack(mask_chosed_list, dim=0), dim=0).values
        self.probes_mask = probes_mask

    def get_features_variance_ranks_visual(self, x, mask_ratio=[0.10, 0.10, 0.10, 0.10], random=False):
        img_size = x.shape[2]

        encoder_new = deepcopy(self.backbone_new.mit)
        encoder_old = deepcopy(self.backbone_old.mit)
        # detach
        for param in encoder_new.parameters():
            param.detach_()
        for param in encoder_old.parameters():
            param.detach_()
        features_new = encoder_new(x)
        features_old = encoder_old(x)
        self.mask_chosed_list = []
        self.uncertain_mask_list = []  # To store uncertain masks
        for i in range(len(features_new)):
            B, C, H, W = features_new[i].shape
            if random:
                # ================= 随机模式：不计算 variance / 不做基于方差的 top-k =================
                num_sel = int(H * W * mask_ratio[i])

                # 用随机数决定被选中的位置
                rand_vals = torch.rand(B, H * W, device=features_new[i].device)
                _, rand_indices = torch.sort(rand_vals, descending=True, dim=1)
                masked_indices = rand_indices[:, :num_sel]

                # 二值 mask
                mask_chosed = torch.zeros((B, H * W), device=features_new[i].device)
                mask_chosed.scatter_(1, masked_indices, 1)
                mask_chosed = mask_chosed.view(B, H, W)

                # 不确定性可视化：用随机得分在被选位置上“着色”（与原逻辑对应 variance.gather）
                uncertainty_mask = torch.zeros((B, H * W), device=features_new[i].device)
                uncertainty_mask.scatter_(1, masked_indices, rand_vals.gather(1, masked_indices))
                uncertainty_mask = uncertainty_mask.view(B, H, W)
            else:
                feature_new = features_new[i].flatten(2).transpose(1, 2)
                feature_old = features_old[i].flatten(2).transpose(1, 2)
                variance = torch.var(feature_new - feature_old, dim=2)
                # variance = torch.mean(variance, dim=2)
                sorted_variance, sorted_indices = torch.sort(variance, descending=True, dim=1)
                top_k = int(sorted_indices.shape[1] * mask_ratio[i])
                masked_indices = sorted_indices[:, :top_k]
                mask_chosed = torch.zeros_like(sorted_variance)
                mask_chosed.scatter_(1, masked_indices, 1)
                mask_chosed = mask_chosed.view(features_new[i].shape[0], features_new[i].shape[2], features_new[i].shape[3])

                # Create the uncertainty mask by selecting regions with the highest variance
                uncertainty_mask = torch.zeros_like(sorted_variance)
                uncertainty_mask.scatter_(1, masked_indices, variance.gather(1, masked_indices))
                uncertainty_mask = uncertainty_mask.view(
                    features_new[i].shape[0], features_new[i].shape[2], features_new[i].shape[3]
                )

            up_mask = F.interpolate(mask_chosed.unsqueeze(1), size=(img_size, img_size), mode="nearest")
            self.mask_chosed_list.append(up_mask)

            # Upsample the uncertainty mask to the image size
            up_uncertainty_mask = F.interpolate(
                uncertainty_mask.unsqueeze(1), size=(img_size, img_size), mode="nearest"
            )
            self.uncertain_mask_list.append(up_uncertainty_mask)

        self.probes_mask = torch.max(torch.stack(self.mask_chosed_list, dim=0), dim=0).values
        self.probes_uncertainty_mask = torch.max(torch.stack(self.uncertain_mask_list, dim=0), dim=0).values

    def get_masked_outputs_new(self, x, mask_chosed_list, mask_token):
        if x.shape[1] == 1:
            # expand img_size to 3x224x224
            x = x.repeat(1, 3, 1, 1)
        H_, W_ = x.shape[-2:]
        mask_chosed_1, mask_chosed_2, mask_chosed_3, mask_chosed_4 = mask_chosed_list
        mask_token_1, mask_token_2, mask_token_3, mask_token_4 = mask_token
        B = x.shape[0]
        out = []

        # stage 1
        x, H, W = self.backbone_new.mit.patch_embed1(x)
        mask_token_1 = mask_token_1.expand(B, x.shape[1], -1).to(x.device)
        unmask = mask_chosed_1.unsqueeze(2).to(x.device)
        x = x * (1 - unmask) + mask_token_1 * unmask
        for i, blk in enumerate(self.backbone_new.mit.block1):
            x = blk(x, H, W)
        x = self.backbone_new.mit.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out.append(x)

        # stage 2
        x, H, W = self.backbone_new.mit.patch_embed2(x)
        mask_token_2 = mask_token_2.expand(B, x.shape[1], -1).to(x.device)
        unmask = mask_chosed_2.unsqueeze(2).to(x.device)
        x = x * (1 - unmask) + mask_token_2 * unmask
        for i, blk in enumerate(self.backbone_new.mit.block2):
            x = blk(x, H, W)
        x = self.backbone_new.mit.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out.append(x)

        # stage 3
        x, H, W = self.backbone_new.mit.patch_embed3(x)
        mask_token_3 = mask_token_3.expand(B, x.shape[1], -1).to(x.device)
        unmask = mask_chosed_3.unsqueeze(2).to(x.device)
        x = x * (1 - unmask) + mask_token_3 * unmask
        for i, blk in enumerate(self.backbone_new.mit.block3):
            x = blk(x, H, W)
        x = self.backbone_new.mit.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out.append(x)

        # stage 4
        x, H, W = self.backbone_new.mit.patch_embed4(x)
        mask_token_4 = mask_token_4.expand(B, x.shape[1], -1).to(x.device)
        unmask = mask_chosed_4.unsqueeze(2).to(x.device)
        x = x * (1 - unmask) + mask_token_4 * unmask
        for i, blk in enumerate(self.backbone_new.mit.block4):
            x = blk(x, H, W)
        x = self.backbone_new.mit.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out.append(x)

        c1, c2, c3, c4 = out

        _c4 = self.backbone_new.decoder.linear_c4(c4).permute(0, 2, 1).reshape(B, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode="bilinear", align_corners=False)

        _c3 = self.backbone_new.decoder.linear_c3(c3).permute(0, 2, 1).reshape(B, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode="bilinear", align_corners=False)

        _c2 = self.backbone_new.decoder.linear_c2(c2).permute(0, 2, 1).reshape(B, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode="bilinear", align_corners=False)

        _c1 = self.backbone_new.decoder.linear_c1(c1).permute(0, 2, 1).reshape(B, -1, c1.shape[2], c1.shape[3])
        _c1 = F.interpolate(_c1, size=c1.size()[2:], mode="bilinear", align_corners=False)

        _c = self.backbone_new.decoder.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.backbone_new.decoder.dropout(_c)
        x = self.backbone_new.decoder.linear_pred(x)
        x = F.interpolate(x, size=(H_, W_), mode="bilinear", align_corners=True)

        outputs_new_masked = softmax(x).clamp(min=1e-08, max=1.0 - 1e-08)

        return outputs_new_masked

    def get_masked_embedding(
        self,
        x,
    ):
        x = x * (1 - self.probes_mask) + self.mask_token * self.probes_mask
        if x.shape[1] == 1:
            # expand img_size to 3x224x224
            x = x.repeat(1, 3, 1, 1)
        H_, W_ = x.shape[-2:]

        B = x.shape[0]
        out = []

        # stage 1
        x, H, W = self.backbone_new.mit.patch_embed1(x)
        for i, blk in enumerate(self.backbone_new.mit.block1):
            x = blk(x, H, W)
        x = self.backbone_new.mit.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out.append(x)

        # stage 2
        x, H, W = self.backbone_new.mit.patch_embed2(x)
        for i, blk in enumerate(self.backbone_new.mit.block2):
            x = blk(x, H, W)
        x = self.backbone_new.mit.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out.append(x)

        # stage 3
        x, H, W = self.backbone_new.mit.patch_embed3(x)
        for i, blk in enumerate(self.backbone_new.mit.block3):
            x = blk(x, H, W)
        x = self.backbone_new.mit.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out.append(x)

        # stage 4
        x, H, W = self.backbone_new.mit.patch_embed4(x)
        for i, blk in enumerate(self.backbone_new.mit.block4):
            x = blk(x, H, W)
        x = self.backbone_new.mit.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out.append(x)

        c1, c2, c3, c4 = out

        _c4 = self.backbone_new.decoder.linear_c4(c4).permute(0, 2, 1).reshape(B, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode="bilinear", align_corners=False)

        _c3 = self.backbone_new.decoder.linear_c3(c3).permute(0, 2, 1).reshape(B, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode="bilinear", align_corners=False)

        _c2 = self.backbone_new.decoder.linear_c2(c2).permute(0, 2, 1).reshape(B, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode="bilinear", align_corners=False)

        _c1 = self.backbone_new.decoder.linear_c1(c1).permute(0, 2, 1).reshape(B, -1, c1.shape[2], c1.shape[3])
        _c1 = F.interpolate(_c1, size=c1.size()[2:], mode="bilinear", align_corners=False)

        _c = self.backbone_new.decoder.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.backbone_new.decoder.dropout(_c)
        x = self.backbone_new.decoder.linear_pred(x)
        x = F.interpolate(x, size=(H_, W_), mode="bilinear", align_corners=True)

        outputs = softmax(x).clamp(min=1e-08, max=1.0 - 1e-08)

        return out, outputs

    def get_embedding_old(
        self,
        x,
    ):
        if x.shape[1] == 1:
            # expand img_size to 3x224x224
            x = x.repeat(1, 3, 1, 1)
        H_, W_ = x.shape[-2:]

        B = x.shape[0]
        out = []

        # stage 1
        x, H, W = self.backbone_old.mit.patch_embed1(x)
        for i, blk in enumerate(self.backbone_old.mit.block1):
            x = blk(x, H, W)
        x = self.backbone_old.mit.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out.append(x)

        # stage 2
        x, H, W = self.backbone_old.mit.patch_embed2(x)
        for i, blk in enumerate(self.backbone_old.mit.block2):
            x = blk(x, H, W)
        x = self.backbone_old.mit.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out.append(x)

        # stage 3
        x, H, W = self.backbone_old.mit.patch_embed3(x)
        for i, blk in enumerate(self.backbone_old.mit.block3):
            x = blk(x, H, W)
        x = self.backbone_old.mit.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out.append(x)

        # stage 4
        x, H, W = self.backbone_old.mit.patch_embed4(x)
        for i, blk in enumerate(self.backbone_old.mit.block4):
            x = blk(x, H, W)
        x = self.backbone_old.mit.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out.append(x)

        c1, c2, c3, c4 = out

        return out


if __name__ == "__main__":
    img_size = 192
    test_img = torch.rand(2, 1, img_size, img_size).cuda()

    model = TKRL(input_shape=(1, img_size, img_size)).cuda()
    model.finish()
    probes_mask = model.get_features_variance_ranks(test_img)

    import matplotlib.pyplot as plt

    plt.imshow(probes_mask[0, 0].cpu().numpy(), cmap="gray")
    plt.colorbar()
    plt.savefig("probes_mask.png")
