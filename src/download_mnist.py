import os
import cv2
import numpy as np
from torchvision import datasets

def prepare_mnist(target_dir="dataset/mnist", samples_per_class=1000):
    """
    Mengunduh dataset MNIST resmi dan menyimpannya sebagai citra PNG
    ke dalam subfolder dataset/mnist/0 s.d. dataset/mnist/9.
    
    Default: 1.000 citra per kelas (total 10.000 citra publik)
    agar seimbang dengan dataset mandiri dan cepat dilatih di CPU/GPU.
    """
    os.makedirs(target_dir, exist_ok=True)
    for d in range(10):
        os.makedirs(os.path.join(target_dir, str(d)), exist_ok=True)

    print("Mengunduh dataset MNIST via torchvision...")
    raw_mnist = datasets.MNIST(root="./dataset/_raw_mnist", train=True, download=True)
    
    counts = {d: 0 for d in range(10)}
    print(f"Menyimpan citra MNIST ke {target_dir} (target: {samples_per_class} per kelas)...")
    
    for idx, (img, label) in enumerate(raw_mnist):
        if counts[label] < samples_per_class:
            img_np = np.array(img)
            # MNIST asli adalah 28x28, digit putih berlatar hitam.
            filename = f"mnist_{label}_{counts[label]:04d}.png"
            filepath = os.path.join(target_dir, str(label), filename)
            cv2.imwrite(filepath, img_np)
            counts[label] += 1
            
        if all(c >= samples_per_class for c in counts.values()):
            break

    print("\n" + "="*45)
    print("HASIL PERSIAPAN DATASET MNIST")
    print("="*45)
    for d in range(10):
        print(f"  Kelas '{d}': {counts[d]} citra tersimpan")
    print("="*45)
    print(f"Total citra MNIST tersimpan: {sum(counts.values())}")
    print(f"Lokasi: {target_dir}")

if __name__ == "__main__":
    prepare_mnist()
