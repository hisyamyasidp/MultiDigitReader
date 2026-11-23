import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image

def preprocess_mandiri_image(img_path, target_size=28):
    """
    Menormalkan citra mandiri (angka gelap di kertas terang) menjadi
    format standar MNIST: 28x28, digit putih berlatar hitam, dan berpusat.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((target_size, target_size), dtype=np.uint8)

    # Invert menggunakan Otsu thresholding
    _, inv = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Isolasi kontur terbesar (menghilangkan noise garis/bintik)
    cnts, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return cv2.resize(inv, (target_size, target_size))

    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros_like(inv)
    cv2.drawContours(mask, [c], -1, 255, -1)
    inv = cv2.bitwise_and(inv, mask)

    x, y, w, h = cv2.boundingRect(c)
    digit_crop = inv[y:y+h, x:x+w]

    # Pasang digit ke dalam kotak 20x20 di dalam kanvas 28x28 (seperti MNIST)
    fit_size = int(target_size * 0.72)
    if h > w:
        new_h = fit_size
        new_w = max(1, int(w * (fit_size / h)))
    else:
        new_w = fit_size
        new_h = max(1, int(h * (fit_size / w)))

    resized = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

    return canvas

class CombinedDigitDataset(Dataset):
    """
    Dataset gabungan Mandiri + MNIST dengan dukungan augmentasi terpisah.
    """
    def __init__(self, samples, transform=None, in_channels=1, img_size=28):
        self.samples = samples  # List of (filepath, label, is_mandiri)
        self.transform = transform
        self.in_channels = in_channels
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, is_mandiri = self.samples[idx]

        if is_mandiri:
            img_np = preprocess_mandiri_image(path, target_size=self.img_size)
        else:
            img_np = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img_np is None:
                img_np = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            elif img_np.shape != (self.img_size, self.img_size):
                img_np = cv2.resize(img_np, (self.img_size, self.img_size))

        # Konversi ke PIL Image untuk torchvision transforms
        pil_img = Image.fromarray(img_np)
        if self.in_channels == 3:
            pil_img = pil_img.convert("RGB")

        if self.transform:
            tensor_img = self.transform(pil_img)
        else:
            to_tensor = transforms.ToTensor()
            tensor_img = to_tensor(pil_img)

        return tensor_img, label

def get_data_loaders(
    mandiri_dir="dataset/mandiri",
    mnist_dir="dataset/mnist",
    batch_size=64,
    img_size=28,
    in_channels=1,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42
):
    """
    Mengumpulkan data mandiri dan publik, membagi (split) ketat
    menjadi Train, Validation, dan Test SEBELUM menerapkan augmentasi.
    """
    all_samples = []

    # 1. Kumpulkan sampel Mandiri
    if os.path.exists(mandiri_dir):
        for d in range(10):
            d_path = os.path.join(mandiri_dir, str(d))
            if os.path.isdir(d_path):
                for f in os.listdir(d_path):
                    if f.endswith(('.jpg', '.png', '.jpeg')):
                        all_samples.append((os.path.join(d_path, f), d, True))

    # 2. Kumpulkan sampel MNIST
    if os.path.exists(mnist_dir):
        for d in range(10):
            d_path = os.path.join(mnist_dir, str(d))
            if os.path.isdir(d_path):
                for f in os.listdir(d_path):
                    if f.endswith(('.jpg', '.png', '.jpeg')):
                        all_samples.append((os.path.join(d_path, f), d, False))

    print(f"Total seluruh sampel terkumpul: {len(all_samples)}")
    mandiri_cnt = sum(1 for s in all_samples if s[2])
    mnist_cnt = sum(1 for s in all_samples if not s[2])
    print(f"  - Sampel Mandiri: {mandiri_cnt}")
    print(f"  - Sampel MNIST  : {mnist_cnt}")

    # 3. SPLIT DATA SECARA KETAT (Stratified berdasarkan Label & Sumber)
    labels = [s[1] for s in all_samples]

    # Split pertama: Pisahkan Train dari (Val + Test)
    train_samples, temp_samples = train_test_split(
        all_samples,
        train_size=train_ratio,
        stratify=labels,
        random_state=random_seed
    )

    # Split kedua: Bagi temp_samples menjadi Val dan Test
    temp_labels = [s[1] for s in temp_samples]
    val_rel_ratio = val_ratio / (val_ratio + test_ratio)
    val_samples, test_samples = train_test_split(
        temp_samples,
        train_size=val_rel_ratio,
        stratify=temp_labels,
        random_state=random_seed
    )

    print("\n" + "="*45)
    print("HASIL PEMBAGIAN DATASET (STRICT SPLIT)")
    print("="*45)
    print(f"Train Set      : {len(train_samples)} sampel ({train_ratio*100:.0f}%) -> [Diberi Augmentasi]")
    print(f"Validation Set : {len(val_samples)} sampel ({val_ratio*100:.0f}%) -> [Murni Tanpa Augmentasi]")
    print(f"Test Set       : {len(test_samples)} sampel ({test_ratio*100:.0f}%) -> [Murni Tanpa Augmentasi]")
    print("="*45)

    # 4. DEFINISI TRANSFORMASI (Augmentasi HANYA untuk Train)
    if in_channels == 1:
        norm = transforms.Normalize((0.1307,), (0.3081,))
    else:
        norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        norm
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        norm
    ])

    train_dataset = CombinedDigitDataset(train_samples, transform=train_transform, in_channels=in_channels, img_size=img_size)
    val_dataset = CombinedDigitDataset(val_samples, transform=eval_transform, in_channels=in_channels, img_size=img_size)
    test_dataset = CombinedDigitDataset(test_samples, transform=eval_transform, in_channels=in_channels, img_size=img_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    t_load, v_load, te_load = get_data_loaders()
    for imgs, lbls in t_load:
        print(f"Batch sample tensor shape: {imgs.shape}, labels: {lbls[:10].tolist()}")
        break
