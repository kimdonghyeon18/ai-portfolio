"""최종 모델(UNetAE)과 공통 함수. 노트북의 정의를 그대로 옮겼습니다."""
import math
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class UNetAE(nn.Module):
    def __init__(self, c=32):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, c, 3, 1, 1), nn.ReLU(),
                                  nn.Conv2d(c, c, 3, 1, 1), nn.ReLU())                 # (B,32,H,W)
        self.enc2 = nn.Sequential(nn.Conv2d(c, 2 * c, 3, 2, 1), nn.ReLU(),
                                  nn.Conv2d(2 * c, 2 * c, 3, 1, 1), nn.ReLU())         # (B,64,H/2,W/2)
        self.enc3 = nn.Sequential(nn.Conv2d(2 * c, 4 * c, 3, 2, 1), nn.ReLU(),
                                  nn.Conv2d(4 * c, 4 * c, 3, 1, 1), nn.ReLU())         # (B,128,H/4,W/4)
        self.mid = nn.Sequential(nn.Conv2d(4 * c, 4 * c, 3, 2, 1), nn.ReLU())          # (B,128,H/8,W/8)
        self.up3 = nn.Sequential(nn.ConvTranspose2d(4 * c, 4 * c, 4, 2, 1), nn.ReLU())
        self.dec3 = nn.Sequential(nn.Conv2d(8 * c, 2 * c, 3, 1, 1), nn.ReLU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(2 * c, 2 * c, 4, 2, 1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.Conv2d(4 * c, c, 3, 1, 1), nn.ReLU())
        self.up1 = nn.Sequential(nn.ConvTranspose2d(c, c, 4, 2, 1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Conv2d(2 * c, c, 3, 1, 1), nn.ReLU())
        self.out = nn.Sequential(nn.Conv2d(c + 1, 1, 3, 1, 1), nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        m = self.mid(e3)
        u3 = self.up3(m)[:, :, :e3.shape[2], :e3.shape[3]]          # 홀수 크기에서 넘친 칸 자르기
        d3 = self.dec3(torch.cat([u3, e3], dim=1))                  # 스킵: 1/4 단계
        u2 = self.up2(d3)[:, :, :e2.shape[2], :e2.shape[3]]
        d2 = self.dec2(torch.cat([u2, e2], dim=1))                  # 스킵: 1/2 단계
        u1 = self.up1(d2)[:, :, :e1.shape[2], :e1.shape[3]]
        d1 = self.dec1(torch.cat([u1, e1], dim=1))                  # 스킵: 원래 크기 단계
        return self.out(torch.cat([d1, x], dim=1))                  # 스킵: 입력 이미지


def load_model(path, device="cpu"):
    model = UNetAE()
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def load_gray(path):
    """흑백 이미지를 0~1 float32 배열로 읽기"""
    return np.array(Image.open(path).convert("L"), dtype=np.float32) / 255


def load_folder(data_dir, name):
    files = sorted(os.listdir(os.path.join(data_dir, name)), key=lambda f: int(f[:-4]))
    return files, [load_gray(os.path.join(data_dir, name, f)) for f in files]


def predict(model, img, device="cpu"):
    with torch.no_grad():
        x = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)   # (1,1,H,W)
        return model(x)[0, 0].cpu().numpy()


def rmse_psnr(preds, targets):
    """모든 이미지의 픽셀을 모아 한 번에 계산"""
    sse = sum(((p - t) ** 2).sum() for p, t in zip(preds, targets))
    n = sum(t.size for t in targets)
    rmse = math.sqrt(sse / n)
    return rmse, 20 * math.log10(1 / rmse)


def split_ids(n, seed=0, n_val=24):
    """노트북과 같은 검증 분리"""
    perm = np.random.default_rng(seed).permutation(n)
    return perm[n_val:], perm[:n_val]
