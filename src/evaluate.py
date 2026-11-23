import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from dataset import get_data_loaders
from model_cnn import get_custom_cnn
from model_mobilenet import get_mobilenet_v2

def evaluate_single_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    latencies = []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            t0 = time.perf_counter()
            outputs = model(imgs)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) / imgs.size(0))
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(lbls.numpy())
            
    avg_latency_ms = np.mean(latencies) * 1000
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0
    return np.array(all_targets), np.array(all_preds), avg_latency_ms, fps

def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*60)
    print(f"EVALUASI DAN KOMPARASI OBJEKTIF MODEL (Device: {device})")
    print("="*60)
    
    _, _, test_loader = get_data_loaders(batch_size=64)
    
    # 1. Muat Model A (Custom CNN)
    cnn_path = "models/best_custom_cnn.pth"
    cnn_model = get_custom_cnn(num_classes=10).to(device)
    if os.path.exists(cnn_path):
        cnn_model.load_state_dict(torch.load(cnn_path, map_location=device))
        print("Model A (Custom CNN) berhasil dimuat.")
    else:
        print(f"File {cnn_path} tidak ditemukan.")
        return

    # 2. Muat Model B (MobileNetV2)
    mob_path = "models/best_mobilenet.pth"
    mob_model = get_mobilenet_v2(num_classes=10).to(device)
    if os.path.exists(mob_path):
        mob_model.load_state_dict(torch.load(mob_path, map_location=device))
        print("Model B (MobileNetV2) berhasil dimuat.")
    else:
        print(f"File {mob_path} tidak ditemukan.")
        return

    # Hitung metrik
    y_true_cnn, y_pred_cnn, lat_cnn, fps_cnn = evaluate_single_model(cnn_model, test_loader, device)
    y_true_mob, y_pred_mob, lat_mob, fps_mob = evaluate_single_model(mob_model, test_loader, device)
    
    acc_cnn = np.mean(y_true_cnn == y_pred_cnn) * 100
    acc_mob = np.mean(y_true_mob == y_pred_mob) * 100
    
    p_cnn, r_cnn, f1_cnn, sup_cnn = precision_recall_fscore_support(y_true_cnn, y_pred_cnn, labels=range(10))
    p_mob, r_mob, f1_mob, sup_mob = precision_recall_fscore_support(y_true_mob, y_pred_mob, labels=range(10))
    
    report_cnn = classification_report(y_true_cnn, y_pred_cnn, digits=4)
    report_mob = classification_report(y_true_mob, y_pred_mob, digits=4)

    # Cetak ke Terminal
    print("\n" + "="*65)
    print("TABEL PERBANDINGAN PERFORMA OBJEKTIF (DATA UJI INDEPENDEN)")
    print("="*65)
    print(f"{'Metrik':<25} | {'Model A (Custom CNN)':<18} | {'Model B (MobileNetV2)':<18}")
    print("-" * 65)
    print(f"{'Akurasi Data Uji':<25} | {acc_cnn:>16.2f}% | {acc_mob:>16.2f}%")
    print(f"{'Macro Avg F1-Score':<25} | {np.mean(f1_cnn):>16.4f}  | {np.mean(f1_mob):>16.4f} ")
    print(f"{'Latensi per Digit':<25} | {lat_cnn:>14.2f} ms | {lat_mob:>14.2f} ms")
    print(f"{'Kecepatan Inferensi':<25} | {fps_cnn:>14.1f} FPS | {fps_mob:>14.1f} FPS")
    print("="*65)
    
    print("\n--- Classification Report Model A (Custom CNN) ---")
    print(report_cnn)
    print("\n--- Classification Report Model B (MobileNetV2) ---")
    print(report_mob)
    
    # Simpan Laporan Teknis ke File TXT agar mudah disalin ke Laporan Kuliah
    report_file = "models/laporan_evaluasi_lengkap.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("="*65 + "\n")
        f.write("HASIL EVALUASI LENGKAP KEDUA MODEL (PROYEK 3)\n")
        f.write("="*65 + "\n\n")
        f.write(f"Akurasi Model A (Custom CNN) : {acc_cnn:.2f}%\n")
        f.write(f"Akurasi Model B (MobileNetV2): {acc_mob:.2f}%\n")
        f.write(f"Latensi Model A : {lat_cnn:.2f} ms ({fps_cnn:.1f} FPS)\n")
        f.write(f"Latensi Model B : {lat_mob:.2f} ms ({fps_mob:.1f} FPS)\n\n")
        f.write("--- CLASSIFICATION REPORT MODEL A (Custom CNN) ---\n")
        f.write(report_cnn + "\n\n")
        f.write("--- CLASSIFICATION REPORT MODEL B (MobileNetV2) ---\n")
        f.write(report_mob + "\n\n")
        f.write("--- RINCIAN PER KELAS (TABEL) ---\n")
        f.write("Kelas | Model A (P / R / F1)          | Model B (P / R / F1)\n")
        f.write("-" * 55 + "\n")
        for d in range(10):
            f.write(f"  {d}   | {p_cnn[d]:.4f} / {r_cnn[d]:.4f} / {f1_cnn[d]:.4f} | {p_mob[d]:.4f} / {r_mob[d]:.4f} / {f1_mob[d]:.4f}\n")
    print(f"\nLaporan evaluasi teks lengkap disimpan di: {report_file}")

    # Plot Confusion Matrix Berdampingan
    cm_cnn = confusion_matrix(y_true_cnn, y_pred_cnn)
    cm_mob = confusion_matrix(y_true_mob, y_pred_mob)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=range(10), yticklabels=range(10))
    axes[0].set_title(f'Confusion Matrix: Custom CNN (Acc: {acc_cnn:.2f}%)')
    axes[0].set_xlabel('Prediksi')
    axes[0].set_ylabel('Label Sebenarnya')
    
    sns.heatmap(cm_mob, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=range(10), yticklabels=range(10))
    axes[1].set_title(f'Confusion Matrix: MobileNetV2 (Acc: {acc_mob:.2f}%)')
    axes[1].set_xlabel('Prediksi')
    axes[1].set_ylabel('Label Sebenarnya')
    
    plt.tight_layout()
    cm_path = "models/confusion_matrix_comparison.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Gambar Confusion Matrix komparasi disimpan di: {cm_path}")

if __name__ == "__main__":
    run_evaluation()
