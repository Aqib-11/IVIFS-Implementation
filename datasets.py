# ivifs/datasets.py
import os, random, torch, numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class VIFSegDataset(Dataset):
    def __init__(self, root, split, img_size=(480,640), dataset_name="FMB"):
        self.root = root
        self.split = split
        self.vis_dir   = os.path.join(root, "visible")
        self.ir_dir    = os.path.join(root, "infrared")
        self.lab_dir   = os.path.join(root, "labels")
        self.files = sorted([f for f in os.listdir(self.vis_dir) if f.endswith((".png",".jpg"))])
        self.resize = T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR)
        self.to_tensor = T.ToTensor()
        self.dataset_name = dataset_name

    def _random_location_mask(self, h, w, prob=0.5):
        # Binary mask; 1=interest area
        m = np.zeros((h, w), dtype=np.uint8)
        if random.random() < prob:
            # random rectangle or polygon
            x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
            x2, y2 = random.randint(x1+1, w-1), random.randint(y1+1, h-1)
            m[y1:y2, x1:x2] = 1
        else:
            m[:,:] = 1  # default all-on
        return torch.from_numpy(m)

    def __getitem__(self, i):
        fname = self.files[i]
        vis = Image.open(os.path.join(self.vis_dir, fname)).convert("RGB")
        ir  = Image.open(os.path.join(self.ir_dir, fname)).convert("L")
        lab = Image.open(os.path.join(self.lab_dir, fname)).convert("L")

        vis = self.to_tensor(self.resize(vis))   # [3,H,W]
        ir  = self.to_tensor(self.resize(ir))    # [1,H,W]
        lab = torch.from_numpy(np.array(self.resize(lab), dtype=np.int64))  # [H,W]

        H, W = lab.shape
        M = self._random_location_mask(H, W)    # [H,W]

        # α provided externally; here we keep 0.5 by default for training curriculum
        return {"visible": vis, "infrared": ir, "label": lab, "mask": M, "alpha": torch.tensor(0.5)}

    def __len__(self): 
        return len(self.files)
