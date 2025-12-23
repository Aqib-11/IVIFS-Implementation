Iimport matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(20, 14), facecolor='white')
ax.set_xlim(0, 20)
ax.set_ylim(0, 16)
ax.axis('off')

# Color scheme
colors = {
    'input': '#E3F2FD',      # Light blue
    'irm': '#F3E5F5',        # Light purple  
    'encoder': '#E8F5E8',    # Light green
    'attention': '#FFF3E0',  # Light orange
    'decoder': '#FFF8E1',    # Light yellow
    'output': '#F1F8E9',     # Very light green
    'loss': '#FFEBEE'        # Light red
}

# 1. TEXT PROMPT
ax.add_patch(FancyBboxPatch((1, 14.5), 3, 1.2, boxstyle="round,pad=0.1", 
                           facecolor=colors['input'], edgecolor='black', lw=2))
ax.text(2.5, 15, 'Text Prompt (T)', ha='center', va='center', fontsize=12, weight='bold')
ax.text(2.5, 14.7, '"focus on person"', ha='center', va='center', fontsize=10)

# 2. IRM Modules
ax.add_patch(FancyBboxPatch((1, 12), 2.5, 1.8, boxstyle="round,pad=0.1", 
                           facecolor=colors['irm'], edgecolor='black', lw=2))
ax.text(2.25, 12.8, 'IFAM', ha='center', va='center', fontsize=11, weight='bold')
ax.text(2.25, 12.5, 'CLIP+MLP', ha='center', fontsize=9)
ax.text(2.25, 12.2, 'β∈{-1,1}, γ∈{0..5}', ha='center', fontsize=9)
ax.text(2.25, 11.9, 'α=0.5+β×0.1γ', ha='center', fontsize=9)

ax.add_patch(FancyBboxPatch((6, 12), 2.5, 1.8, boxstyle="round,pad=0.1", 
                           facecolor=colors['irm'], edgecolor='black', lw=2))
ax.text(7.25, 12.8, 'ISMG', ha='center', va='center', fontsize=11, weight='bold')
ax.text(7.25, 12.5, 'LISA/SAM', ha='center', fontsize=9)
ax.text(7.25, 12.2, 'Mv, Mi → Max(M)', ha='center', fontsize=9)
ax.text(7.25, 11.9, 'M[H,W] ∈{0,1}', ha='center', fontsize=9)

# 3. INPUT IMAGES
ax.add_patch(FancyBboxPatch((1, 9.5), 2.2, 1.2, boxstyle="round,pad=0.1", 
                           facecolor=colors['input'], edgecolor='black', lw=2))
ax.text(2.1, 10, 'I_vis', ha='center', va='center', fontsize=11, weight='bold')
ax.text(2.1, 9.7, '[3,H,W] RGB', ha='center', fontsize=9)

ax.add_patch(FancyBboxPatch((6, 9.5), 2.2, 1.2, boxstyle="round,pad=0.1", 
                           facecolor=colors['input'], edgecolor='black', lw=2))
ax.text(7.1, 10, 'I_ir', ha='center', va='center', fontsize=11, weight='bold')
ax.text(7.1, 9.7, '[1,H,W] Thermal', ha='center', fontsize=9)

# 4. DUAL ENCODER
ax.add_patch(FancyBboxPatch((2.5, 6.5), 5.5, 2.2, boxstyle="round,pad=0.1", 
                           facecolor=colors['encoder'], edgecolor='black', lw=2))
ax.text(5.25, 8, 'DualEncoder', ha='center', va='center', fontsize=14, weight='bold')
ax.text(5.25, 7.6, 'n_stages=4, base_c=32', ha='center', fontsize=11)
ax.text(5.25, 7.2, 'Cs=[32,64,128,256]', ha='center', fontsize=11)
ax.text(2.8, 6.9, 'E_vi Stages', ha='center', fontsize=9)
ax.text(7.7, 6.9, 'E_ir Stages', ha='center', fontsize=9)

# 5. MODAL INTERACTION
ax.add_patch(FancyBboxPatch((10, 7.5), 3.5, 1.2, boxstyle="round,pad=0.1", 
                           facecolor=colors['attention'], edgecolor='black', lw=2))
ax.text(11.75, 8.1, 'α-aware ModalInteraction', ha='center', va='center', fontsize=12, weight='bold')
ax.text(11.75, 7.9, '(Eq.1)', ha='center', fontsize=10)
ax.text(11.75, 7.7, 'S-Att → C-Att → G-Att', ha='center', fontsize=10)

# 6. MULTI-SCALE FEATURES
ax.add_patch(FancyBboxPatch((2, 4.5), 2.5, 1, boxstyle="round,pad=0.1", 
                           facecolor=colors['encoder'], edgecolor='black', lw=2))
ax.text(3.25, 5, 'feats_vis', ha='center', va='center', fontsize=11, weight='bold')
ax.text(3.25, 4.7, '[4,Cs,H/2^i,W/2^i]', ha='center', fontsize=9)

ax.add_patch(FancyBboxPatch((7, 4.5), 2.5, 1, boxstyle="round,pad=0.1", 
                           facecolor=colors['encoder'], edgecolor='black', lw=2))
ax.text(8.25, 5, 'feats_ir', ha='center', va='center', fontsize=11, weight='bold')
ax.text(8.25, 4.7, '[4,Cs,H/2^i,W/2^i]', ha='center', fontsize=9)

# 7. DECODERS
ax.add_patch(FancyBboxPatch((1, 2.5), 3, 1.5, boxstyle="round,pad=0.1", 
                           facecolor=colors['decoder'], edgecolor='black', lw=2))
ax.text(2.5, 3.5, 'FusionDecoder', ha='center', va='center', fontsize=12, weight='bold')
ax.text(2.5, 3.2, 'D_f (Eq.2)', ha='center', fontsize=10)
ax.text(2.5, 2.9, '1×1→Upsample→3×3→RGB', ha='center', fontsize=9)

ax.add_patch(FancyBboxPatch((9, 2.5), 3, 1.5, boxstyle="round,pad=0.1", 
                           facecolor=colors['decoder'], edgecolor='black', lw=2))
ax.text(10.5, 3.5, 'SegDecoder', ha='center', va='center', fontsize=12, weight='bold')
ax.text(10.5, 3.2, 'D_s (Eq.6,7)', ha='center', fontsize=10)
ax.text(10.5, 2.9, 'MaskAware→3×3→Head→Logits', ha='center', fontsize=9)

# 8. OUTPUTS
ax.add_patch(FancyBboxPatch((1.5, 1), 2.5, 1, boxstyle="round,pad=0.1", 
                           facecolor=colors['output'], edgecolor='black', lw=2))
ax.text(2.75, 1.5, 'I_f [B,3,H,W]', ha='center', va='center', fontsize=11, weight='bold')
ax.text(2.75, 1.2, '(sigmoid [0,1])', ha='center', fontsize=9)

ax.add_patch(FancyBboxPatch((9.5, 1), 2.5, 1, boxstyle="round,pad=0.1", 
                           facecolor=colors['output'], edgecolor='black', lw=2))
ax.text(10.75, 1.5, 'Logits [B,15,H,W]', ha='center', va='center', fontsize=11, weight='bold')

# 9. LOSSES
ax.add_patch(FancyBboxPatch((1, -0.3), 2.5, 0.8, boxstyle="round,pad=0.1", 
                           facecolor=colors['loss'], edgecolor='black', lw=2))
ax.text(2.25, 0.1, 'L_fusion', ha='center', va='center', fontsize=11, weight='bold')
ax.text(2.25, -0.1, 'αD_vis+(1-α)D_ir', ha='center', fontsize=9)

ax.add_patch(FancyBboxPatch((9.5, -0.3), 2.5, 0.8, boxstyle="round,pad=0.1", 
                           facecolor=colors['loss'], edgecolor='black', lw=2))
ax.text(10.75, 0.1, 'L_seg (CE)', ha='center', va='center', fontsize=11, weight='bold')
ax.text(10.75, -0.1, 'MaskWeighted (Eq.7)', ha='center', fontsize=9)

# TOTAL LOSS
ax.add_patch(FancyBboxPatch((5.5, -0.8), 3.5, 0.5, boxstyle="round,pad=0.1", 
                           facecolor='#FFCDD2', edgecolor='red', lw=3))
ax.text(7.25, -0.55, 'L_total = L_f + L_s', ha='center', va='center', fontsize=14, weight='bold', color='red')

# MASK CONNECTION
ax.add_patch(ConnectionPatch((8.25, 12), (10, 3.2), "data", "data", 
                            arrowstyle="->", lw=2, color="purple", linestyle="--"))

# ALPHA CONNECTION  
con_alpha = ConnectionPatch((3.75, 11.8), (11, 7.9), "data", "data", 
                           arrowstyle="->", lw=2, color="orange", linestyle="--")
ax.add_artist(con_alpha)

plt.title('CVIFSM Architecture - Complete Implementation\n145M params | 169G FLOPs | T4×2 Stable\nAqib Javed Reproduction', 
          fontsize=16, weight='bold', pad=30, color='darkblue')

plt.tight_layout()
plt.savefig('cvifsm_architecture_complete.png', dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
plt.show()

