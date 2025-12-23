# ivifs/models/attention.py
import torch, torch.nn as nn, torch.nn.functional as F

class ChannelAtt(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Flatten(),
                                 nn.Linear(c, c//r),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(c//r, c))
    
    def forward(self, x):
        w = torch.sigmoid(self.mlp(x))
        return x * w.unsqueeze(-1).unsqueeze(-1)

class SpatialAtt(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, 1, kernel_size=7, padding=3)
    
    def forward(self, x):
        w = torch.sigmoid(self.conv(x))
        return x * w

class GateAtt(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.g = nn.Sequential(nn.Conv2d(c, c, 1), nn.Sigmoid())
    
    def forward(self, x_v, x_i):
        g = self.g(torch.cat([x_v, x_i], dim=1))
        return x_v * g[:, :x_v.size(1)] + x_i * g[:, x_v.size(1):]

class ModalInteraction(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c_att = ChannelAtt(c)
        self.s_att = SpatialAtt(c)
        self.g_att = GateAtt(c*2)
    
    def forward(self, f_v, f_i, alpha: float):
        # α-aware scaling (Eq. 1): Att_i( α·F^i_vis , (1-α)·F^i_ir )
        f_vs = self.c_att(self.s_att(alpha * f_v))
        f_is = self.c_att(self.s_att((1 - alpha) * f_i))
        # gated cross-modal mixing
        f_v_next = self.g_att(f_vs, f_is)
        f_i_next = self.g_att(f_is, f_vs)
        return f_v_next, f_i_next
