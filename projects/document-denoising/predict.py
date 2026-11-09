"""학습한 가중치로 검증 점수 확인, 이미지 복원, 제출 파일 만들기.

python predict.py --data denoising --eval                  # 검증 24장 RMSE
python predict.py --data denoising --input test --out out  # 폴더의 이미지를 복원해 PNG로 저장
python predict.py --data denoising --submission sub.csv    # Kaggle 제출 파일
"""
import argparse
import os

import numpy as np
from PIL import Image

from model import load_folder, load_model, predict, rmse_psnr, split_ids


def write_submission(files, preds, path):
    with open(path, "w") as fp:
        fp.write("id,value\n")
        for f in sorted(files, key=lambda f: f[:-4]):           # "1", "10", "100" ... 문자열 순서
            pred = preds[files.index(f)]
            H, W = pred.shape
            for col in range(W):
                for row in range(H):                            # 행이 먼저 바뀜
                    fp.write(f"{f[:-4]}_{row + 1}_{col + 1},{pred[row, col]:.4f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="denoising")
    parser.add_argument("--weights", default="weights/unet_ae_final.pt")
    parser.add_argument("--eval", action="store_true", help="검증 24장으로 RMSE 계산")
    parser.add_argument("--input", help="복원할 이미지 폴더 이름 (예: test)")
    parser.add_argument("--out", default="restored")
    parser.add_argument("--submission", help="제출 CSV 경로")
    args = parser.parse_args()

    model = load_model(args.weights)

    if args.eval:
        _, noisy = load_folder(args.data, "train")
        _, clean = load_folder(args.data, "train_cleaned")
        _, val_idx = split_ids(len(noisy))
        base, _ = rmse_psnr([noisy[i] for i in val_idx], [clean[i] for i in val_idx])
        rmse, psnr = rmse_psnr([predict(model, noisy[i]) for i in val_idx], [clean[i] for i in val_idx])
        print(f"검증 RMSE {rmse:.4f}  PSNR {psnr:.2f} dB  (손상본 그대로 {base:.4f})")

    if args.input or args.submission:
        name = args.input or "test"
        files, imgs = load_folder(args.data, name)
        preds = [predict(model, img) for img in imgs]
        if args.input:
            os.makedirs(args.out, exist_ok=True)
            for f, p in zip(files, preds):
                Image.fromarray((p * 255).round().astype(np.uint8)).save(os.path.join(args.out, f))
            print(f"{len(files)}장 저장: {args.out}/")
        if args.submission:
            write_submission(files, preds, args.submission)
            print("제출 파일:", args.submission)


if __name__ == "__main__":
    main()
