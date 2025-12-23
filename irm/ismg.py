class ISMG:
    def __init__(self, lisa_model):
        self.model = lisa_model  # pre-trained LISA (or SAM+caption grounding as fallback)
    
    def generate_mask(self, I_vis, I_ir, text):
        Mv = self.model.segment(I_vis, text)  # [H,W] float in [0,1]
        Mi = self.model.segment(I_ir,  text)  # [H,W] float in [0,1]
        Mv = (Mv > 0.5).astype("uint8")
        Mi = (Mi > 0.5).astype("uint8")
        M = np.maximum(Mv, Mi)  # Eq. (10)
        return torch.from_numpy(M)  # [H,W] in {0,1}
