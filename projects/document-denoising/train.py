"""UNetAE 학습. 예: python train.py --data denoising --epochs 600"""
import argparse
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import UNetAE, load_folder, predict, rmse_psnr, split_ids

CROP = 256


class CropPairs(Dataset):
    """손상본과 원본을 같은 위치에서 256×256으로 잘라 돌려줌"""
    def __init__(self, noisy, clean, ids):
        self.noisy, self.clean, self.ids = noisy, clean, ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        x = torch.tensor(self.noisy[self.ids[k]])
        y = torch.tensor(self.clean[self.ids[k]])
        H, W = x.shape
        top = torch.randint(0, H - CROP + 1, (1,)).item()
        left = torch.randint(0, W - CROP + 1, (1,)).item()
        x = x[top:top + CROP, left:left + CROP]
        y = y[top:top + CROP, left:left + CROP]
        return x.unsqueeze(0), y.unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="denoising", help="train, train_cleaned 폴더가 있는 경로")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out", default="unet_ae_final.pt")
    args = parser.parse_args()

    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, noisy = load_folder(args.data, "train")
    _, clean = load_folder(args.data, "train_cleaned")
    tr_idx, val_idx = split_ids(len(noisy), seed)
    print(f"device {device} / 학습 {len(tr_idx)}장 / 검증 {len(val_idx)}장")

    model = UNetAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(CropPairs(noisy, clean, tr_idx), batch_size=8, shuffle=True)
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0 or epoch == args.epochs:
            model.eval()
            rmse, psnr = rmse_psnr([predict(model, noisy[i], device) for i in val_idx],
                                   [clean[i] for i in val_idx])
            print(f"epoch {epoch:4d}  val RMSE {rmse:.4f}  PSNR {psnr:.2f} dB  ({time.time() - start:.0f}s)")

    torch.save(model.state_dict(), args.out)
    print("저장:", args.out)


if __name__ == "__main__":
    main()
