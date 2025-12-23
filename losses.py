import torch
import torch.nn as nn
import torch.nn.functional as F

def gradient_loss(x, y):
    def grad(t):
        gx = t[..., :, 1:] - t[..., :, :-1]
        gy = t[..., 1:, :] - t[..., :-1, :]
        return gx, gy
    gx1, gy1 = grad(x); gx2, gy2 = grad(y)
    return (gx1 - gx2).abs().mean() + (gy1 - gy2).abs().mean()

class ControllableFusionLoss(nn.Module):
    def __init__(self, w_l1=1.0, w_grad=1.0):
        super().__init__()
        self.w_l1, self.w_grad = w_l1, w_grad
    
    def forward(self, I_f, I_vis, I_ir, alpha):
        # Downsample IR to 3-ch for similarity or stack channel for fair gradient measurement
        I_ir_rgb = I_ir.repeat(1,3,1,1)
        D_vis = self.w_l1 * (I_f - I_vis).abs().mean() + self.w_grad * gradient_loss(I_f, I_vis)
        D_ir  = self.w_l1 * (I_f - I_ir_rgb).abs().mean() + self.w_grad * gradient_loss(I_f, I_ir_rgb)
        return alpha * D_vis + (1 - alpha) * D_ir

class MaskWeightedCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction="none")  # per-pixel
    
    def forward(self, logits, labels, M):
        # Eq. (7): mask-weighted CE
        # M: [B,H,W] in {0,1}
        ce = self.ce(logits, labels)  # [B,H,W]
        return (ce * M.float()).sum() / (M.float().sum() + 1e-6)
