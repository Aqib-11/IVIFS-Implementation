import torch, torch.nn as nn

# Use open_clip or transformers CLIP text encoder as frozen interpreter
class IFAM(nn.Module):
    def __init__(self, txt_dim=512):
        super().__init__()
        self.text_encoder = FrozenCLIPTextEncoder(output_dim=txt_dim)  # wrap pre-trained CLIP text
        self.cls_beta = nn.Sequential(nn.Linear(txt_dim, 128), nn.ReLU(), nn.Linear(128, 2))   # {-1,1}
        self.cls_gamma = nn.Sequential(nn.Linear(txt_dim, 128), nn.ReLU(), nn.Linear(128, 6))  # {0..5}
    
    def forward(self, tokenized_text):
        t = self.text_encoder(tokenized_text)  # [B, txt_dim]
        p_beta  = torch.softmax(self.cls_beta(t), dim=-1)    # [B,2]
        p_gamma = torch.softmax(self.cls_gamma(t), dim=-1)   # [B,6]
        beta_idx  = p_beta.argmax(-1)  # 0 or 1
        beta = torch.where(beta_idx==1, torch.tensor(1.0, device=t.device), torch.tensor(-1.0, device=t.device))
        gamma = p_gamma.argmax(-1).float()                   # 0..5
        conf_beta = p_beta.max(-1).values
        # Eq. (8): confidence gate
        beta = torch.where(conf_beta > 0.5, beta, torch.zeros_like(beta))
        alpha = 0.5 + beta * (0.1 * gamma)
        alpha = alpha.clamp(0.0, 1.0)
        return alpha, p_beta, p_gamma
