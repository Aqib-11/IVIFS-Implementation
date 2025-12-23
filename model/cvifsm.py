# ivifs/models/cvifsm.py
import torch.nn as nn

class CVIFSM(nn.Module):
    def __init__(self, n_stages=4, base_c=32, n_classes=15):
        super().__init__()
        Cs = [base_c*(2**i) for i in range(n_stages)]
        self.encoder = DualEncoder(n_stages=n_stages, base_c=base_c)
        self.fusion_dec = FusionDecoder(Cs)
        self.seg_dec    = SegDecoder(Cs, n_classes)
    
    def forward(self, I_vis, I_ir, alpha, M):
        feats_vis, feats_ir = self.encoder(I_vis, I_ir, alpha)
        I_f   = self.fusion_dec(feats_vis, feats_ir, alpha)
        logits = self.seg_dec(feats_vis, feats_ir, M)
        return I_f, logits
