import os
import cv2
import numpy as np

def extract_and_organize_dataset(
    source_dir='Digits-DigitMandiri.coco/train',
    target_base_dir='dataset/mandiri',
    video_prefix='WIN_20261123_10_44_53_Pro_mp4',
    padding=10
):
    """
    Mengekstrak digit 0-9 dari frame video berkualitas tinggi dan
    mengorganisasikannya ke dalam subfolder dataset/mandiri/0 s.d. 9.
    """
    os.makedirs(target_base_dir, exist_ok=True)
    for d in range(10):
        os.makedirs(os.path.join(target_base_dir, str(d)), exist_ok=True)
        
    all_files = sorted([f for f in os.listdir(source_dir) if f.startswith(video_prefix)])
    print(f"Total frame yang dianalisis ({video_prefix}): {len(all_files)}")
    
    extracted_counts = {str(d): 0 for d in range(10)}
    valid_frames_count = 0
    
    for fn in all_files:
        img_path = os.path.join(source_dir, fn)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
        )
        
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digits = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            # Filter kandidat bounding box digit
            if 40 <= h <= 150 and 15 <= w <= 150 and area > 500:
                digits.append((x, y, w, h))
                
        # Syarat: tepat terdeteksi 10 objek digit (0-9)
        if len(digits) == 10:
            # Urutkan berdasarkan koordinat X dari kiri ke kanan
            digits = sorted(digits, key=lambda b: b[0])
            
            # Verifikasi jarak konsisten antar digit berurutan
            diffs = [digits[i+1][0] - digits[i][0] for i in range(9)]
            if all(30 <= d <= 120 for d in diffs):
                valid_frames_count += 1
                base_name = fn.split('.')[0]
                
                # Simpan masing-masing digit ke kelasnya (0 s.d. 9)
                for digit_val, (x, y, w, h) in enumerate(digits):
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(img.shape[1], x + w + padding)
                    y2 = min(img.shape[0], y + h + padding)
                    
                    crop = img[y1:y2, x1:x2]
                    out_path = os.path.join(
                        target_base_dir, 
                        str(digit_val), 
                        f"digit_{digit_val}_{base_name}.jpg"
                    )
                    cv2.imwrite(out_path, crop)
                    extracted_counts[str(digit_val)] += 1

    print("\n" + "="*45)
    print("HASIL EKSTRAKSI DATASET MANDIRI")
    print("="*45)
    print(f"Frame valid & terverifikasi : {valid_frames_count} frame")
    print(f"Total digit berhasil diekstrak : {sum(extracted_counts.values())} citra")
    print("-" * 45)
    for d in range(10):
        print(f"  Kelas '{d}' : {extracted_counts[str(d)]} sampel")
    print("="*45)
    return extracted_counts

if __name__ == '__main__':
    extract_and_organize_dataset()
