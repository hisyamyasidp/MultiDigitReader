import os
import time
from collections import deque, Counter
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model_cnn import get_custom_cnn
from model_mobilenet import get_mobilenet_v2

class RealTimeDigitRecognizer:
    def __init__(self, cnn_path="models/best_custom_cnn.pth", mob_path="models/best_mobilenet.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Menjalankan Real-Time Digit Recognizer pada perangkat: {self.device}")
        
        # Inisialisasi model
        self.models = {}
        if os.path.exists(cnn_path):
            m_cnn = get_custom_cnn(num_classes=10).to(self.device)
            m_cnn.load_state_dict(torch.load(cnn_path, map_location=self.device))
            m_cnn.eval()
            self.models['Custom CNN'] = m_cnn
            print("Model Custom CNN berhasil dimuat.")
        else:
            print(f"Perhatian: {cnn_path} tidak ditemukan.")

        if os.path.exists(mob_path):
            m_mob = get_mobilenet_v2(num_classes=10).to(self.device)
            m_mob.load_state_dict(torch.load(mob_path, map_location=self.device))
            m_mob.eval()
            self.models['MobileNetV2'] = m_mob
            print("Model MobileNetV2 berhasil dimuat.")
        else:
            print(f"Perhatian: {mob_path} tidak ditemukan.")

        if not self.models:
            raise RuntimeError("Tidak ada model terlatih yang ditemukan di folder models/! Silakan jalankan script train terlebih dahulu.")

        self.current_model_name = list(self.models.keys())[0]
        self.model = self.models[self.current_model_name]
        
        # Buffer untuk Temporal Smoothing (Majority Voting)
        # Menyimpan sequence hasil prediksi dari 7 frame terakhir
        self.sequence_history = deque(maxlen=7)
        self.stable_sequence = ""

    def switch_model(self):
        names = list(self.models.keys())
        idx = names.index(self.current_model_name)
        new_idx = (idx + 1) % len(names)
        self.current_model_name = names[new_idx]
        self.model = self.models[self.current_model_name]
        self.sequence_history.clear()
        print(f"Beralih ke model: {self.current_model_name}")

    def preprocess_digit_crop(self, crop):
        """
        Menormalkan crop digit mentah menjadi tensor input (1, 1, 28, 28)
        dengan mempertahankan aspect ratio dan padding tengah (format MNIST).
        """
        h, w = crop.shape
        fit_size = 20
        if h > w:
            new_h = fit_size
            new_w = max(1, int(w * (fit_size / h)))
        else:
            new_w = fit_size
            new_h = max(1, int(h * (fit_size / w)))

        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((28, 28), dtype=np.uint8)
        y_off = (28 - new_h) // 2
        x_off = (28 - new_w) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

        # Normalisasi PyTorch MNIST
        tensor = torch.from_numpy(canvas).float().unsqueeze(0).unsqueeze(0) / 255.0
        tensor = (tensor - 0.1307) / 0.3081
        return tensor.to(self.device)

    def run(self, camera_id=1):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Gagal membuka kamera ID {camera_id}. Coba ID lain (misal 1).")
            return

        print("\n" + "="*50)
        print("APLIKASI VISI KOMPUTER REAL-TIME MULTI-DIGIT")
        print("Kontrol:")
        print("  - Tekan 'm' : Ganti arsitektur model (Custom CNN <-> MobileNetV2)")
        print("  - Tekan 'c' : Reset / bersihkan temporal buffer")
        print("  - Tekan 'q' : Keluar dari aplikasi")
        print("="*50)

        prev_time = time.time()
        fps = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / (curr_time - prev_time + 1e-6))
            prev_time = curr_time

            h_frame, w_frame = frame.shape[:2]

            # Zona ROI persegi di tengah layar
            roi_w, roi_h = int(w_frame * 0.70), int(h_frame * 0.45)
            roi_x = (w_frame - roi_w) // 2
            roi_y = (h_frame - roi_h) // 2
            roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

            # Prapemrosesan ROI
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray_roi, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
            )

            # Deteksi kontur digit di dalam ROI
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            digit_boxes = []

            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                area = cv2.contourArea(c)
                # Filter dimensi digit yang valid (mengabaikan garis kertas atau noise kecil)
                if 30 <= h <= (roi_h - 10) and 12 <= w <= (roi_w // 2) and area > 250:
                    digit_boxes.append((x, y, w, h))

            # Filter multi-digit: batas 2 hingga 4 digit per spesifikasi proyek
            current_frame_sequence = ""
            confidences = []

            if 1 <= len(digit_boxes) <= 6:
                # Wajib: Pengurutan dari kiri ke kanan berdasarkan koordinat X
                digit_boxes = sorted(digit_boxes, key=lambda b: b[0])

                with torch.no_grad():
                    for (x, y, w, h) in digit_boxes:
                        # Ekstrak biner dari crop digit
                        crop_binary = thresh[y:y+h, x:x+w]
                        input_tensor = self.preprocess_digit_crop(crop_binary)
                        
                        outputs = self.model(input_tensor)
                        probs = F.softmax(outputs, dim=1)
                        conf, pred = torch.max(probs, 1)
                        
                        digit_str = str(pred.item())
                        conf_val = conf.item() * 100
                        
                        current_frame_sequence += digit_str
                        confidences.append(conf_val)

                        # Gambar Bounding Box per digit
                        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(
                            roi, f"{digit_str} ({conf_val:.0f}%)",
                            (x, max(15, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
                        )

            # Mekanisme Temporal Smoothing (Majority Voting)
            if current_frame_sequence:
                self.sequence_history.append(current_frame_sequence)
                # Ambil sequence yang paling sering muncul dari buffer
                most_common = Counter(self.sequence_history).most_common(1)[0][0]
                self.stable_sequence = most_common
            else:
                if len(self.sequence_history) > 0:
                    self.sequence_history.popleft()
                if len(self.sequence_history) == 0:
                    self.stable_sequence = "-"

            # Gambar visualisasi ROI dan Header Info pada layar
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 200, 0), 2)
            
            # Panel Informasi Atas
            info_bar = np.zeros((70, w_frame, 3), dtype=np.uint8)
            cv2.putText(
                info_bar, f"Model: {self.current_model_name} (Tekan 'm' ganti)", 
                (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
            )
            cv2.putText(
                info_bar, f"FPS: {fps:.1f}", 
                (w_frame - 150, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
            )
            
            avg_conf = np.mean(confidences) if confidences else 0.0
            cv2.putText(
                info_bar, f"Hasil Sequence: {self.stable_sequence}  | Conf: {avg_conf:.1f}%", 
                (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            
            combined = np.vstack([info_bar, frame])
            cv2.imshow("Handwritten Multi-Digit Recognition (Proyek 3)", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                self.switch_model()
            elif key == ord('c'):
                self.sequence_history.clear()
                self.stable_sequence = "-"

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = RealTimeDigitRecognizer()
    recognizer.run()
