"""Gradio 데모: 손상된 문서 이미지를 올리면 복원 결과를 보여줌. 실행: python app.py"""
import gradio as gr
import numpy as np
from PIL import Image

from model import load_model, predict

MAX_SIDE = 1600          # 너무 큰 이미지는 CPU 메모리와 시간을 위해 줄여서 처리
model = load_model("weights/unet_ae_final.pt")


def denoise(image):
    if image is None:
        raise gr.Error("이미지를 올려 주세요.")
    img = image.convert("L")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    out = predict(model, np.array(img, dtype=np.float32) / 255)
    return Image.fromarray((out * 255).round().astype(np.uint8))


demo = gr.Interface(
    fn=denoise,
    inputs=gr.Image(type="pil", label="손상된 문서 이미지"),
    outputs=gr.Image(type="pil", label="복원 결과"),
    examples=["examples/sample.png"],
    title="문서 이미지 노이즈 제거 (U-Net 오토인코더)",
    description="얼룩·구김이 있는 흑백 문서 이미지를 깨끗하게 복원합니다. "
                "Kaggle Denoising Dirty Documents로 학습했고 검증 RMSE 0.0128, Kaggle 점수 0.01837입니다.",
)

if __name__ == "__main__":
    demo.launch()
