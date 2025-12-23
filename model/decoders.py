# ivifs/models/decoders.py (continued)
import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionDecoder(nn.Module):
    def __init__(self, Cs):
        super().__init__()
        # simple FPN-like upsampling + conv
        self.upconvs = nn.ModuleList([nn.Conv2d(c, Cs[0], 1) for c in Cs])
        self.fuse = nn.Sequential(
            nn.Conv2d(len(Cs)*Cs[0], Cs[0], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(Cs[0], 3, 1)  # RGB fused output
        )
    
    def forward(self, feats_vis, feats_ir, alpha):
        ups = []
        for fv, fi, up in zip(feats_vis, feats_ir, self.upconvs):
            # Eq. (2): D_f( α·F^i_vis, (1-α)·F^i_ir, ... )
            m = torch.cat([alpha * fv, (1 - alpha) * fi], dim=1)
            u = up(F.interpolate(m, size=feats_vis[0].shape[-2:], mode="bilinear", align_corners=False))
            ups.append(u)
        x = torch.cat(ups, dim=1)
        return torch.sigmoid(self.fuse(x))  # [B,3,H,W] in [0,1]

class MaskAware(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c+1, c, 3, padding=1)
    
    def forward(self, f, M):
        # M: [B,H,W] -> [B,1,H,W]
        m = M.unsqueeze(1).float()
        return self.conv(torch.cat([f * m, m], dim=1))
    
class SegDecoder(nn.Module):
    def __init__(self, Cs, n_classes):
        super().__init__()
        self.mask_mods = nn.ModuleList([MaskAware(c) for c in Cs])
        self.seg_head = nn.Sequential(
            nn.Conv2d(Cs[0]*len(Cs), Cs[0], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(Cs[0], n_classes, 1)
        )
    
    def forward(self, feats_vis, feats_ir, M):
        ups = []
        for (fv, fi, mm) in zip(feats_vis, feats_ir, self.mask_mods):
            f = mm((fv + fi) / 2, F.interpolate(M.unsqueeze(1), size=fv.shape[-2:], mode="nearest").squeeze(1))
            u = F.interpolate(f, size=feats_vis[0].shape[-2:], mode="bilinear", align_corners=False)
            ups.append(u)
        x = torch.cat(ups, dim=1)
        return self.seg_head(x)  # logits [B,C,H,W]
