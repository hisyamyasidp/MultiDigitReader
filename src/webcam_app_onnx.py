import os
import time
from collections import deque, Counter
import cv2
import numpy as np
import onnxruntime as ort

class ONNXRealTimeDigitRecognizer:
    """
    Aplikasi Inferensi Real-Time Multi-Digit Berbasis ONNX Runtime murni.
    Berjalan sangat ringan dan cepat di CPU tanpa memerlukan pustaka PyTorch.
    Memenuhi syarat Poin Bonus Optimasi Deployment (+5% s.d. +10%).
    """
    def __init__(self, cnn_onnx="models/custom_cnn.onnx", mob_onnx="models/mobilenet.onnx"):
        print("Menginisialisasi ONNX Runtime Inference Engine (CPUExecutionProvider)...")
        self.sessions = {}
        
        # Load Model A (Custom CNN ONNX)
        if os.path.exists(cnn_onnx):
            sess_cnn = ort.InferenceSession(cnn_onnx, providers=['CPUExecutionProvider'])
            self.sessions['Custom CNN (ONNX)'] = {
                'session': sess_cnn,
                'input_name': sess_cnn.get_inputs()[0].name,
                'output_name': sess_cnn.get_outputs()[0].name
            }
            print("Model Custom CNN (ONNX) berhasil dimuat.")
        else:
            print(f"Perhatian: {cnn_onnx} tidak ditemukan.")

        # Load Model B (MobileNetV2 ONNX)
        if os.path.exists(mob_onnx):
            sess_mob = ort.InferenceSession(mob_onnx, providers=['CPUExecutionProvider'])
            self.sessions['MobileNetV2 (ONNX)'] = {
                'session': sess_mob,
                'input_name': sess_mob.get_inputs()[0].name,
                'output_name': sess_mob.get_outputs()[0].name
            }
            print("Model MobileNetV2 (ONNX) berhasil dimuat.")
        else:
            print(f"Perhatian: {mob_onnx} tidak ditemukan.")

        if not self.sessions:
            raise RuntimeError("Tidak ada model .onnx di folder models/! Silakan jalankan export_onnx.py terlebih dahulu.")

        self.current_model_name = list(self.sessions.keys())[0]
        self.current_sess_info = self.sessions[self.current_model_name]
        
        # Buffer untuk Temporal Smoothing (Majority Voting)
        self.sequence_history = deque(maxlen=7)
        self.stable_sequence = "-"

    def switch_model(self):
        names = list(self.sessions.keys())
        idx = names.index(self.current_model_name)
        new_idx = (idx + 1) % len(names)
        self.current_model_name = names[new_idx]
        self.current_sess_info = self.sessions[self.current_model_name]
        self.sequence_history.clear()
        print(f"Beralih ke model: {self.current_model_name}")

    def preprocess_digit_crop(self, crop):
        """
        Menormalkan crop digit mentah menjadi array numpy (1, 1, 28, 28)
        dengan mempertahankan aspect ratio dan padding tengah (format MNIST).
        Murni menggunakan OpenCV dan NumPy tanpa PyTorch.
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
        canvas = np.zeros((28, 28), dtype=np.float32)
        y_off = (28 - new_h) // 2
        x_off = (28 - new_w) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

        # Normalisasi MNIST standar (mean=0.1307, std=0.3081)
        canvas = (canvas / 255.0 - 0.1307) / 0.3081
        
        # Bentuk tensor: (1, 1, 28, 28)
        tensor_np = np.expand_dims(np.expand_dims(canvas, axis=0), axis=0).astype(np.float32)
        return tensor_np

    def predict_digit(self, input_np):
        """
        Inferensi cepat via ONNX Runtime
        """
        session = self.current_sess_info['session']
        input_name = self.current_sess_info['input_name']
        
        # Eksekusi ONNX
        logits = session.run(None, {input_name: input_np})[0][0]
        
        # Softmax murni NumPy
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        pred_label = int(np.argmax(probs))
        confidence = float(probs[pred_label]) * 100.0
        return pred_label, confidence

    def run(self, camera_id=1):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Gagal membuka kamera ID {camera_id}. Coba ID lain (misal 1).")
            return

        print("\n" + "="*55)
        print("APLIKASI REAL-TIME MULTI-DIGIT (ONNX RUNTIME ENGINE)")
        print("Kontrol:")
        print("  - Tekan 'm' : Ganti model (Custom CNN ONNX <-> MobileNetV2 ONNX)")
        print("  - Tekan 'c' : Reset / bersihkan temporal buffer")
        print("  - Tekan 'q' : Keluar dari aplikasi")
        print("="*55)

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
                # Filter dimensi digit yang valid
                if 30 <= h <= (roi_h - 10) and 12 <= w <= (roi_w // 2) and area > 250:
                    digit_boxes.append((x, y, w, h))

            current_frame_sequence = ""
            confidences = []

            if 1 <= len(digit_boxes) <= 6:
                # Wajib: Pengurutan dari kiri ke kanan berdasarkan koordinat X
                digit_boxes = sorted(digit_boxes, key=lambda b: b[0])

                for (x, y, w, h) in digit_boxes:
                    crop_binary = thresh[y:y+h, x:x+w]
                    input_np = self.preprocess_digit_crop(crop_binary)
                    
                    pred_label, conf_val = self.predict_digit(input_np)
                    digit_str = str(pred_label)
                    
                    current_frame_sequence += digit_str
                    confidences.append(conf_val)

                    # Visualisasi Bounding Box per digit
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        roi, f"{digit_str} ({conf_val:.0f}%)",
                        (x, max(15, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
                    )

            # Mekanisme Temporal Smoothing (Majority Voting)
            if current_frame_sequence:
                self.sequence_history.append(current_frame_sequence)
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
                info_bar, f"Engine: ONNX Runtime | {self.current_model_name} (Tekan 'm')", 
                (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
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
            cv2.imshow("Handwritten Multi-Digit Recognition (ONNX Deployment)", combined)

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
    recognizer = ONNXRealTimeDigitRecognizer()
    recognizer.run()
