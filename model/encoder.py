ivifs/models/encoders.py
2     class ConvStage(nn.Module):
3         def __init__(self, in_c, out_c, stride=2):
4             super().__init__()
5             self.block = nn.Sequential(
6                 nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1),
7                 nn.BatchNorm2d(out_c),
8                 nn.ReLU(inplace=True),
9                 nn.Conv2d(out_c, out_c, 3, padding=1),
10                 nn.BatchNorm2d(out_c),
11                 nn.ReLU(inplace=True),
12             )
13         def forward(self, x): return self.block(x)
14     
15     class DualEncoder(nn.Module):
16         def __init__(self, n_stages=4, base_c=32):
17             super().__init__()
18             Cs = [base_c*(2**i) for i in range(n_stages)]
19             self.vis_stages = nn.ModuleList([ConvStage(3 if i==0 else Cs[i-1], Cs[i]) for i in range(n_stages)])
20             self.ir_stages  = nn.ModuleList([ConvStage(1 if i==0 else Cs[i-1], Cs[i]) for i in range(n_stages)])
21             self.att_blocks = nn.ModuleList([ModalInteraction(Cs[i]) for i in range(n_stages)])
22     
23         def forward(self, I_vis, I_ir, alpha):
24             f_vis, f_ir = I_vis, I_ir
25             feats_vis, feats_ir = [], []
26             for i, (sv, si, att) in enumerate(zip(self.vis_stages, self.ir_stages, self.att_blocks)):
27                 f_vis = sv(f_vis)
28                 f_ir  = si(f_ir)
29                 f_vis, f_ir = att(f_vis, f_ir, alpha)
30                 feats_vis.append(f_vis)
31                 feats_ir.append(f_ir)
32             return feats_vis, feats_ir
