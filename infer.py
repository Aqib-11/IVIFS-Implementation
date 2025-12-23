def interactive_infer(cvifsm, ifam, ismg, I_vis, I_ir, text=None, max_iters=3):
    device = next(cvifsm.parameters()).device
    alpha = torch.tensor([0.5], device=device)
    M = torch.ones(I_vis.shape[-2:], dtype=torch.uint8, device=device)  # all-on
    
    for _ in range(max_iters):
        I_f, logits = cvifsm(I_vis.to(device), I_ir.to(device), alpha, M)
        seg = logits.argmax(1)  # [B,H,W]
        
        if text is None:
            break
            
        # update via IRM (Interactive Refinement Module)
        alpha, _, _ = ifam(tokenize(text).to(device))
        M = ismg.generate_mask(I_vis[0].cpu(), I_ir[0].cpu(), text).to(device)
    
    return I_f, seg
