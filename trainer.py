# ivifs/trainer.py (excerpt)
def train_cvifsm(model, loader, optimizer, device, n_classes):
    fusion_loss = ControllableFusionLoss()
    seg_loss    = MaskWeightedCELoss()
    model.train()
    
    for batch in loader:
        I_vis = batch["visible"].to(device)
        I_ir  = batch["infrared"].to(device)
        labels= batch["label"].to(device)
        M     = batch["mask"].to(device)
        alpha = batch["alpha"].to(device)  # 0.5 default during training curriculum

        I_f, logits = model(I_vis, I_ir, alpha, M)
        Lf = fusion_loss(I_f, I_vis, I_ir, alpha)
        Ls = seg_loss(logits, labels, M)
        loss = Lf + Ls  # L_total = L_fusion + L_seg
        
        optimizer.zero_grad(); loss.backward(); optimizer.step()
def train_ifam(ifam, prompt_loader, optimizer, device):
    ifam.train()
    
    for batch in prompt_loader:
        toks = batch["tokens"].to(device)
        y_beta = batch["y_beta"].to(device)   # {0,1}
        y_gamma= batch["y_gamma"].to(device)  # {0..5}
        
        alpha, p_beta, p_gamma = ifam(toks)
        loss = ifam_loss(p_beta, y_beta, p_gamma, y_gamma)
        
        optimizer.zero_grad(); loss.backward(); optimizer.step()
