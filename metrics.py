# ivifs/metrics.py
import torch, math

def entropy(img):  # img in [0,1], [B,3,H,W]
    # approximate 256-bin entropy per image channel then average
    b = 256
    hist = torch.histc(img.flatten(), bins=b, min=0.0, max=1.0)
    p = hist / hist.sum().clamp_min(1)
    p = p[p>0]
    return float((-p * p.log()).sum() / math.log(b))

def std_dev(img):  # per-channel then mean
    return float(img.std())

def corrcoef(a, b):  # average CC vs visible & infrared
    a = a.flatten(); b = b.flatten()
    a = a - a.mean(); b = b - b.mean()
    denom = (a.std() * b.std()).clamp_min(1e-9)
    return float((a*b).mean() / denom)

def fusion_metrics(I_f, I_vis, I_ir):
    EN = entropy(I_f)
    SD = std_dev(I_f)
    CC_vis = corrcoef(I_f, I_vis)
    CC_ir  = corrcoef(I_f, I_ir.repeat(1,3,1,1))
    CC = 0.5*(CC_vis + CC_ir)
    return {"EN": EN, "SD": SD, "CC": CC}
