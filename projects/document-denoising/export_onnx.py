"""가중치를 브라우저용 ONNX로 변환하고, PyTorch 결과와 같은지 확인. 실행: python export_onnx.py"""
import numpy as np
import onnxruntime as ort
import torch

from model import load_model

model = load_model("weights/unet_ae_final.pt")
torch.onnx.export(
    model, torch.rand(1, 1, 258, 540), "web/model.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {2: "H", 3: "W"}, "output": {2: "H", 3: "W"}},   # 어떤 크기든 받도록
    opset_version=17, dynamo=False,
)

session = ort.InferenceSession("web/model.onnx")
for h, w in [(258, 540), (420, 540), (333, 777)]:            # 홀수 크기도 확인
    x = np.random.rand(1, 1, h, w).astype(np.float32)
    onnx_out = session.run(None, {"input": x})[0]
    with torch.no_grad():
        torch_out = model(torch.tensor(x)).numpy()
    diff = np.abs(onnx_out - torch_out).max()
    print(f"{h}x{w}: 출력 {onnx_out.shape}, 최대 차이 {diff:.2e}")
    assert onnx_out.shape == (1, 1, h, w) and diff < 1e-4
print("OK")
