import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import get_data_loaders
from model_cnn import get_custom_cnn

def train_model_a(epochs=30, batch_size=64, lr=0.001):
    os.makedirs("models", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Memulai Pelatihan Model A (Custom CNN) pada perangkat: {device}")
    
    # 1. Dapatkan DataLoader (Split murni dilakukan di dalam dataset.py)
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=batch_size)
    
    # 2. Inisialisasi Model, Loss, Optimizer, Scheduler
    model = get_custom_cnn(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_path = "models/best_custom_cnn.pth"
    
    start_total = time.time()
    for epoch in range(1, epochs + 1):
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)
            
        train_loss = running_loss / total
        train_acc = correct / total
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                val_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == lbls).sum().item()
                val_total += lbls.size(0)
                
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch}/{epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Model terbaik disimpan! (Val Acc: {val_acc*100:.2f}%)")

    total_time = time.time() - start_total
    print("\n" + "="*50)
    print(f"Pelatihan Selesai dalam {total_time:.1f} detik.")
    print(f"Model terbaik disimpan di: {best_model_path}")
    print(f"Akurasi Validasi Tertinggi: {best_val_acc*100:.2f}%")
    print("="*50)
    
    # 3. Plot Kurva Pelatihan
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history['train_loss'], label='Train Loss')
    plt.plot(range(1, epochs + 1), history['val_loss'], label='Val Loss')
    plt.title('Custom CNN - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history['train_acc'], label='Train Acc')
    plt.plot(range(1, epochs + 1), history['val_acc'], label='Val Acc')
    plt.title('Custom CNN - Akurasi')
    plt.xlabel('Epoch')
    plt.ylabel('Akurasi')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = "models/training_curve_cnn.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Kurva pelatihan disimpan di: {plot_path}")

if __name__ == "__main__":
    train_model_a(epochs=30, batch_size=64, lr=0.001)
