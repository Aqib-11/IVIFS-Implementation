class FusionDecoder(nn.Module):
3         def __init__(self, Cs):
4             super().__init__()
5             # simple FPN-like upsampling + conv
6             self.upconvs = nn.ModuleList([nn.Conv2d(c, Cs[0], 1) for c in Cs])
7             self.fuse = nn.Sequential(
8                 nn.Conv2d(len(Cs)*Cs[0], Cs[0], 3, padding=1),
9                 nn.ReLU(inplace=True),
10                 nn.Conv2d(Cs[0], 3, 1)  # RGB fused output
11             )
12     
13         def forward(self, feats_vis, feats_ir, alpha):
14             ups = []
15             for fv, fi, up in zip(feats_vis, feats_ir, self.upconvs):
16                 # Eq. (2): D_f( α·F^i_vis, (1-α)·F^i_ir, ... )
17                 m = torch.cat([alpha * fv, (1 - alpha) * fi], dim=1)
18                 u = up(F.interpolate(m, size=feats_vis[0].shape[-2:], mode="bilinear", align_corners=False))
19                 ups.append(u)
20             x = torch.cat(ups, dim=1)
21             return torch.sigmoid(self.fuse(x))  # [B,3,H,W] in [0,1]
