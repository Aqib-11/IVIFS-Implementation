# ivifs/run_train_cvifsm.py
import torch, argparse
from torch.utils.data import DataLoader
from ivifs.models.cvifsm import CVIFSM
from ivifs.datasets import VIFSegDataset
from ivifs.trainer import train_cvifsm

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["FMB","MSRS"], required=True)
    ap.add_argument("--root", type=str, default="data/FMB")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--batch", type=int, default=3)
    args = ap.parse_args()

    n_classes = 15 if args.dataset=="FMB" else 9
    ds = VIFSegDataset(args.root, split="train", dataset_name=args.dataset)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4)

    model = CVIFSM(n_stages=4, base_c=32, n_classes=n_classes).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_cvifsm(model, dl, opt, device="cuda", n_classes=n_classes)
