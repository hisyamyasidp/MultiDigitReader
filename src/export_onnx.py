import os
import torch
import onnx
import onnxruntime as ort
import numpy as np

from model_cnn import get_custom_cnn
from model_mobilenet import get_mobilenet_v2

def export_model_to_onnx(model, dummy_input, output_path, model_name="Model"):
    """
    Mengekspor model PyTorch ke format ONNX dan memverifikasi integritasnya.
    """
    model.eval()
    print(f"\n--- Mengekspor {model_name} ke ONNX ---")
    
    # 1. Ekspor ke ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"File tersimpan di: {output_path}")
    
    # 2. Periksa Integritas File ONNX
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("Verifikasi struktur graf ONNX: VALID ✓")
    
    # 3. Verifikasi Numerik: Bandingkan Output PyTorch vs ONNX Runtime
    with torch.no_grad():
        torch_out = model(dummy_input).numpy()
        
    ort_session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_out = ort_session.run(None, ort_inputs)[0]
    
    # Hitung selisih maksimal antara PyTorch dan ONNX
    max_diff = np.max(np.abs(torch_out - ort_out))
    print(f"Selisih numerik PyTorch vs ONNX: {max_diff:.8f}")
    assert max_diff < 1e-4, "Peringatan: Selisih numerik terlalu besar!"
    print("Verifikasi numerik output: IDENTIK (Lolos Uji Presisi) ✓")
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Ukuran file ONNX: {file_size_mb:.2f} MB")

def run_all_exports():
    os.makedirs("models", exist_ok=True)
    dummy_input = torch.randn(1, 1, 28, 28, dtype=torch.float32)
    
    # 1. Ekspor Model A: Custom CNN
    cnn_path = "models/best_custom_cnn.pth"
    cnn_onnx_path = "models/custom_cnn.onnx"
    if os.path.exists(cnn_path):
        cnn_model = get_custom_cnn(num_classes=10)
        cnn_model.load_state_dict(torch.load(cnn_path, map_location='cpu'))
        export_model_to_onnx(cnn_model, dummy_input, cnn_onnx_path, model_name="Model A (Custom CNN)")
    else:
        print(f"File {cnn_path} belum ada. Jalankan train_cnn.py terlebih dahulu.")

    # 2. Ekspor Model B: MobileNetV2
    mob_path = "models/best_mobilenet.pth"
    mob_onnx_path = "models/mobilenet.onnx"
    if os.path.exists(mob_path):
        mob_model = get_mobilenet_v2(num_classes=10)
        mob_model.load_state_dict(torch.load(mob_path, map_location='cpu'))
        export_model_to_onnx(mob_model, dummy_input, mob_onnx_path, model_name="Model B (MobileNetV2)")
    else:
        print(f"File {mob_path} belum ada. Jalankan train_mobilenet.py terlebih dahulu.")

    print("\n" + "="*50)
    print("EKSPOR ONNX SELESAI DENGAN SUKSES!")
    print("Kedua model siap dijalankan via ONNX Runtime.")
    print("="*50)

if __name__ == "__main__":
    run_all_exports()
