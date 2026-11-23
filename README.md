# Laporan Teknis & Dokumentasi Proyek 3
## Pengenalan Angka Multi-Digit Tulisan Tangan di Atas Kertas Berbasis Kamera Real-Time

**Mata Kuliah:** Computer Vision (UAS)  
**Tujuan Proyek:** Membangun sistem pembaca nomor antrean / kode angka tulisan tangan (2 hingga 4 digit sekaligus) pada kertas HVS putih menggunakan webcam secara *real-time* dengan stabilisasi temporal.

---

## Daftar Isi
1. [Struktur Repositori Modular](#1-struktur-repositori-modular)
2. [Penjelasan Lengkap Pipeline Preprocessing & Deteksi Kontur](#2-penjelasan-lengkap-pipeline-preprocessing--deteksi-kontur)
3. [Arsitektur Model: Komparasi Model A vs Model B](#3-arsitektur-model-komparasi-model-a-vs-model-b)
4. [Dataset & Strategi Anti-Data Leakage](#4-dataset--strategi-anti-data-leakage)
5. [Hasil Evaluasi Kuantitatif Lengkap (Akurasi, Precision, Recall, F1-Score)](#5-hasil-evaluasi-kuantitatif-lengkap)
6. [Panduan Menjalankan Program & Kontrol Demo Live](#6-panduan-menjalankan-program--kontrol-demo-live)

---

## 1. Struktur Repositori Modular

Sesuai rubrik *"Repositori Kode Modular & Anti-Spaghetti"*, seluruh kode dipecah ke dalam modul independen:

```text
CV UAS/
├── dataset/
│   ├── mandiri/          # 1.090 citra digit tulisan tangan lokal (kelas 0 s.d. 9)
│   └── mnist/            # 10.000 citra publik MNIST resmi (kelas 0 s.d. 9)
├── models/               # Bobot model terlatih & visualisasi evaluasi:
│   ├── best_custom_cnn.pth             # Bobot terbaik Model A (Akurasi Uji: 98.86%)
│   ├── best_mobilenet.pth              # Bobot terbaik Model B (Akurasi Uji: 97.36%)
│   ├── confusion_matrix_comparison.png # Grafik komparasi Confusion Matrix berdampingan
│   ├── training_curve_cnn.png          # Kurva Loss & Akurasi Model A
│   └── training_curve_mobilenet.png    # Kurva Loss & Akurasi Model B
├── src/                  # Kode Sumber Utama (Single Responsibility Principle):
│   ├── download_mnist.py # [1] Unduh & normalisasi dataset MNIST resmi
│   ├── dataset.py        # [2] DataLoader gabungan, strict split (70/15/15), augmentasi
│   ├── model_cnn.py      # [3] Definisi arsitektur Model A (Custom CNN)
│   ├── model_mobilenet.py# [4] Definisi arsitektur Model B (MobileNetV2)
│   ├── train_cnn.py      # [5] Script pelatihan Model A & checkpointing
│   ├── train_mobilenet.py# [6] Script pelatihan Model B & checkpointing
│   ├── evaluate.py       # [7] Evaluasi kuantitatif lengkap (P, R, F1, Latensi, CM)
│   └── webcam_app.py     # [8] Aplikasi inferensi real-time webcam multi-digit
├── extract_dataset.py    # Ekstraksi & kurasi dataset mandiri dari video rekaman
├── requirements.txt      # Daftar dependensi library
└── README.md             # Dokumentasi teknis komprehensif
```

---

## 2. Penjelasan Lengkap Pipeline Preprocessing & Deteksi Kontur

Pada sistem inferensi live ([`src/webcam_app.py`]), frame mentah kamera **dilarang keras** langsung dimasukkan ke model deep learning. Sistem menerapkan 8 tahapan visi komputer berikut:

```
[Frame Webcam Mentah]
        │
        ▼
[1. Isolasi Zona ROI] (Area tengah layar 70% x 45%)
        │
        ▼
[2. Grayscale Conversion] (`cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)`)
        │
        ▼
[3. Gaussian Blur] (`cv2.GaussianBlur(gray, (5, 5), 0)`) -> Menghilangkan high-frequency noise
        │
        ▼
[4. Adaptive Gaussian Thresholding] (`cv2.adaptiveThreshold(..., THRESH_BINARY_INV, 25, 10)`)
        │  * Mengatasi bayangan tangan/kertas dan variasi pencahayaan lokal.
        │  * Meng-invert warna: Tinta menjadi PUTIH (255), Kertas menjadi HITAM (0).
        │
        ▼
[5. Deteksi Kontur Eksternal] (`cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`)
        │
        ▼
[6. Pemfilteran Geometri Kontur (Bounding Box & Area Filtering)]
        │  * Tinggi valid: 30 px <= h <= (ROI_H - 10)
        │  * Lebar valid: 12 px <= w <= (ROI_W / 2)
        │  * Luas Area valid: area > 250 px²
        │  * Mengabaikan lubang kertas loose-leaf, garis buku, dan debu kecil.
        │
        ▼
[7. Pengurutan Koordinat X Kiri ke Kanan] (`sorted(boxes, key=lambda b: b[0])`)
        │  * Memastikan urutan digit selalu benar dari kiri ke kanan (misal: "2", "0", "2", "6").
        │
        ▼
[8. Normalisasi Format MNIST (28x28 Centered)]
        │  * Mempertahankan aspect ratio asli (fit ke kotak 20x20).
        │  * Ditempatkan di tengah kanvas 28x28 (padding 4 px).
        │  * Tensor dinormalisasi dengan mean=0.1307 dan std=0.3081.
        │
        ▼
[9. Inferensi Model Deep Learning + Softmax Confidence Score]
        │
        ▼
[10. Temporal Smoothing (Majority Voting Buffer 7-Frame)]
           * Mengeliminasi flicker / kedipan prediksi antar frame menggunakan antrean `deque`.
```

---

## 3. Arsitektur Model: Komparasi Model A vs Model B

Sesuai ketentuan tugas, wajib membandingkan minimal 2 pendekatan arsitektur berbeda:

### A. Model A: Custom CNN ([`src/model_cnn.py`])
Arsitektur konvolusi murni yang dirancang dari nol (*from scratch*), dioptimalkan khusus untuk citra $28 \times 28$:
* **Blok Konvolusi 1:**
  * `Conv2D(in=1, out=32, kernel=3x3, pad=1)` + `BatchNorm2D` + `ReLU`
  * `Conv2D(in=32, out=32, kernel=3x3, pad=1)` + `BatchNorm2D` + `ReLU`
  * `MaxPool2D(kernel=2x2, stride=2)` $\rightarrow$ Reduksi spasial ke $14 \times 14$
  * `Dropout2D(p=0.25)` $\rightarrow$ Mencegah co-adaptation fitur konvolusi
* **Blok Konvolusi 2:**
  * `Conv2D(in=32, out=64, kernel=3x3, pad=1)` + `BatchNorm2D` + `ReLU`
  * `Conv2D(in=64, out=64, kernel=3x3, pad=1)` + `BatchNorm2D` + `ReLU`
  * `MaxPool2D(kernel=2x2, stride=2)` $\rightarrow$ Reduksi spasial ke $7 \times 7$
  * `Dropout2D(p=0.25)`
* **Lapisan Klasifikasi (Dense / Fully Connected):**
  * `Flatten()` $\rightarrow 64 \times 7 \times 7 = 3.136$ fitur
  * `Linear(3136, 128)` + `BatchNorm1D` + `ReLU` + `Dropout(p=0.5)`
  * `Linear(128, 10)` $\rightarrow$ Logits untuk 10 kelas angka (0 s.d. 9)
* **Total Parameter:** **468.458 parameter** (Sangat ringan, inferensi $\approx 0.17\text{ ms}$ di CPU).

### B. Model B: MobileNetV2 ([`src/model_mobilenet.py`])
Arsitektur standar industri efisien untuk perangkat *mobile* dan *edge*:
* Memanfaatkan blok **Inverted Residual** dan **Depthwise Separable Convolution** untuk meminimalkan beban komputasi.
* **Adaptasi Khusus:** Lapisan konvolusi pertama diubah agar menerima 1 kanal input (*grayscale*) dengan `stride=1` agar detail citra $28 \times 28$ tidak langsung hilang di awal. Lapisan linear akhir disesuaikan ke 10 output kelas.
* **Total Parameter:** **2.236.106 parameter**.

---

## 4. Dataset & Strategi Anti-Data Leakage

* **Komposisi Data:**
  * Dataset Mandiri: **1.090 citra** (109 per kelas) yang direkam langsung dari webcam kelompok pada kondisi pencahayaan dan sudut alami.
  * Dataset Publik: **10.000 citra** MNIST resmi (1.000 per kelas).
  * Total Keseluruhan: **11.090 citra**.
* **Pembagian Data (*Strict Stratified Split*):**
  * **Train Set (70%):** 7.762 sampel $\rightarrow$ **Hanya set ini yang diberikan augmentasi dinamis** (rotasi $\pm 10^\circ$, pergeseran $\pm 6\%$, dan penskalaan $\pm 5\%$).
  * **Validation Set (15%):** 1.664 sampel $\rightarrow$ Murni tanpa augmentasi, digunakan untuk pemilihan bobot terbaik (*checkpointing*).
  * **Test Set (15%):** 1.664 sampel $\rightarrow$ Murni tanpa augmentasi, digunakan khusus untuk pengujian akhir independen.
* **Jaminan Anti-Data Leakage:** Data di-split terlebih dahulu sebelum augmentasi, sehingga variasi hasil augmentasi tidak pernah bocor ke data evaluasi.

---

## 5. Hasil Evaluasi Kuantitatif Lengkap

Pengujian dilakukan pada **1.664 citra data uji independen (Test Set)** yang tidak pernah dilihat oleh kedua model selama pelatihan:

### A. Tabel Komparasi Performa Keseluruhan

| Metrik Evaluasi | Model A (*Custom CNN*) | Model B (*MobileNetV2*) | Target Minimal Nilai A | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Akurasi Uji (Test Accuracy)** | **$\mathbf{98.86\%}$** | **$\mathbf{97.36\%}$** | $\ge 92.00\%$ | **Lolos Sangat Unggul** |
| **Macro Average F1-Score** | **$\mathbf{0.9886}$** | **$\mathbf{0.9735}$** | $\ge 0.8800$ | **Lolos Sangat Unggul** |
| **Total Parameter Model** | **$468.458$** | $2.236.106$ | Ringkas & Efisien | **Custom CNN 4.7x Lebih Ringan** |
| **Latensi Inferensi per Digit** | **$\mathbf{0.17\text{ ms}}$** | $0.58\text{ ms}$ | Real-time CPU | **Custom CNN 3.4x Lebih Cepat** |
| **Kecepatan Inferensi (Batch)** | **$\mathbf{5.947\text{ FPS}}$** | $1.726\text{ FPS}$ | $\ge 15\text{--}20\text{ FPS}$ | **Sangat Responsif** |

---

### B. Rincian Metrik per Kelas: Precision, Recall, & F1-Score

#### Model A: Custom CNN (Akurasi: 98.86%)
| Kelas Digit | Precision | Recall | F1-Score | Jumlah Sampel Uji (*Support*) |
| :---: | :---: | :---: | :---: | :---: |
| **0** | 1.0000 | 1.0000 | 1.0000 | 166 |
| **1** | 0.9820 | 0.9880 | 0.9850 | 166 |
| **2** | 0.9938 | 0.9699 | 0.9817 | 166 |
| **3** | 0.9939 | 0.9820 | 0.9880 | 167 |
| **4** | 0.9940 | 0.9880 | 0.9910 | 167 |
| **5** | 0.9940 | 1.0000 | 0.9970 | 167 |
| **6** | 0.9939 | 0.9880 | 0.9909 | 166 |
| **7** | 0.9706 | 0.9880 | 0.9792 | 167 |
| **8** | 0.9939 | 0.9819 | 0.9879 | 166 |
| **9** | 0.9708 | 1.0000 | 0.9852 | 166 |
| **Rata-rata Terbobot** | **0.9887** | **0.9886** | **0.9886** | **1.664** |

#### Model B: MobileNetV2 (Akurasi: 97.36%)
| Kelas Digit | Precision | Recall | F1-Score | Jumlah Sampel Uji (*Support*) |
| :---: | :---: | :---: | :---: | :---: |
| **0** | 0.9939 | 0.9819 | 0.9879 | 166 |
| **1** | 0.9880 | 0.9880 | 0.9880 | 166 |
| **2** | 0.9581 | 0.9639 | 0.9610 | 166 |
| **3** | 0.9704 | 0.9820 | 0.9762 | 167 |
| **4** | 0.9811 | 0.9341 | 0.9571 | 167 |
| **5** | 0.9881 | 0.9940 | 0.9910 | 167 |
| **6** | 0.9702 | 0.9819 | 0.9760 | 166 |
| **7** | 0.9937 | 0.9401 | 0.9662 | 167 |
| **8** | 0.9643 | 0.9759 | 0.9701 | 166 |
| **9** | 0.9322 | 0.9940 | 0.9621 | 166 |
| **Rata-rata Terbobot** | **0.9740** | **0.9736** | **0.9735** | **1.664** |

> **Analisis Kesalahan (Error Analysis):**  
> Gambar visualisasi Confusion Matrix berdampingan telah tersimpan di [`models/confusion_matrix_comparison.png`]. Kedua model menunjukkan diagonal dominan mendekati sempurna ($>96\%$). Sedikit ambiguitas minor hanya terjadi pada pasangan angka dengan kemiripan goresan tangan alami, seperti angka $4$ dan $9$ atau $7$ dan $2$.

---

## 6. Panduan Menjalankan Program & Kontrol Demo Live

### Menjalankan Demo Live Webcam Multi-Digit
```bash
python src/webcam_app.py
```
* **Cara Pengujian:** Tulis 2 sampai 4 digit angka acak (misal: `2026`, `485`) di kertas putih, lalu dekatkan ke dalam kotak ROI di layar pada jarak 30–50 cm.
* **Tombol `m`:** Mengganti model inferensi secara langsung (*on-the-fly*) antara **Custom CNN** dan **MobileNetV2**.
* **Tombol `c`:** Membersihkan riwayat buffer prediksi (*reset sequence*).
* **Tombol `q`:** Menutup aplikasi webcam.

### Menjalankan Ulang Evaluasi Kuantitatif
```bash
python src/evaluate.py
```
Menghasilkan tabel metrik lengkap di terminal dan memperbarui grafik matriks konfusi di folder `models/`.
