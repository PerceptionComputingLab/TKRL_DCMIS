import torch.nn as nn
import torch
from thop import profile, clever_format



import torch.nn.functional as F
from mp.models.segmentation.mix_transformer import (
    mit_b0,
    mit_b1,
    mit_b2,
    mit_b3,
    mit_b4,
    mit_b5,
)
from mp.models.segmentation.segformer_head import SegFormerHead

class SegFormer(nn.Module):
    def __init__(self, num_classes=2, phi="b0", pretrained=True):
        super(SegFormer, self).__init__()
        self.inchannels = {
            "b0": [32, 64, 160, 256],
            "b1": [64, 128, 320, 512],
            "b2": [64, 128, 320, 512],
            "b3": [64, 128, 320, 512],
            "b4": [64, 128, 320, 512],
            "b5": [64, 128, 320, 512],
        }[phi]
        self.mit = {
            "b0": mit_b0(),
            "b1": mit_b1(),
            "b2": mit_b2(),
            "b3": mit_b3(),
            "b4": mit_b4(),
            "b5": mit_b5(),
        }[phi]
        self.embedding_dim = {
            "b0": 256,
            "b1": 256,
            "b2": 768,
            "b3": 768,
            "b4": 768,
            "b5": 768,
        }[phi]
        self.decoder = SegFormerHead(
            num_classes=num_classes,
            in_channels=self.inchannels,
            embedding_dim=self.embedding_dim,
        )
        self.mit.init_weights("pretrained/mit_b0.pth")

    def forward(self, x):
        H, W = x.shape[-2:]
        x = self.mit(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True)
        return x
    
def count_flops_segformer_b0(h=192, w=192, bs=1, num_classes=2,
                             device="cuda", pretrained=False):
    model = SegFormer(num_classes=num_classes, phi="b0", pretrained=pretrained).to(device).eval()

    # 3 通道输入（若是灰度图，可把单通道重复成 3 通道）
    x = torch.randn(bs, 1, h, w, device=device)

    # 1) Encoder FLOPs/Params
    with torch.no_grad():
        macs_enc, params_enc = profile(model.mit, inputs=(x,), verbose=False)
    flops_enc, params_enc = clever_format([2 * macs_enc, params_enc], "%.3f")

    # 2) Decoder FLOPs/Params（用 encoder 的真实特征做输入）
    with torch.no_grad():
        feats = model.mit(x)  # (c1, c2, c3, c4)

    class _DecOnly(nn.Module):
        def __init__(self, dec): super().__init__(); self.dec = dec
        def forward(self, c1, c2, c3, c4): return self.dec((c1, c2, c3, c4))

    dec_only = _DecOnly(model.decoder).to(device).eval()
    with torch.no_grad():
        macs_dec, params_dec = profile(dec_only, inputs=feats, verbose=False)
    flops_dec, params_dec = clever_format([2 * macs_dec, params_dec], "%.3f")

    # 3) 全模型（包含最后上采样到原图）
    with torch.no_grad():
        macs_all, params_all = profile(model, inputs=(x,), verbose=False)
    flops_all, params_all = clever_format([2 * macs_all, params_all], "%.3f")

    print(f"[Input ] bs={bs}, size={h}x{w}, device={device}")
    print(f"[Encoder] FLOPs={flops_enc}, Params={params_enc}")
    print(f"[Decoder] FLOPs={flops_dec}, Params={params_dec}")
    print(f"[Total ]  FLOPs={flops_all}, Params={params_all}")

    return {
        "enc": {"macs": macs_enc, "flops": 2 * macs_enc, "params": params_enc},
        "dec": {"macs": macs_dec, "flops": 2 * macs_dec, "params": params_dec},
        "all": {"macs": macs_all, "flops": 2 * macs_all, "params": params_all},
    }


if __name__ == "__main__":


    count_flops_segformer_b0(h=192, w=192, bs=1, num_classes=2, device="cuda")
    exit()
        



    from torchinfo import summary

    model = SegFormer(num_classes=2, phi="b0").to("cuda:1")
    summary(model, input_size=(1, 3, 192, 192))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    print(device)
    image = torch.randn(2, 3, 192, 192).to(device)

    model = SegFormer(num_classes=2, phi="b0").to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print("total_params:", total_params)

    out = model(image)
    print(out.shape)




