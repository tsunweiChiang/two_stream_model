import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import matplotlib.pyplot as plt
import torch
from model.model import TwoStreamModel
from tools.data_loader import build_train_val_loaders
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import numpy as np
# 路徑設定
SPATIAL_ROOT  = '/project/Research/two_stream_model/dataset/spatial'
TEMPORAL_ROOT = '/project/Research/two_stream_model/dataset/temporal'
BATCH_SIZE    = 2
EPOCHS        = 100
CKPT_DIR      = Path('/project/Research/two_stream_model/checkpoints')
CKPT_DIR.mkdir(exist_ok=True)

# dataloader
train_loader, val_loader = build_train_val_loaders(
    SPATIAL_ROOT, TEMPORAL_ROOT,
    val_ratio=0.2,
    batch_size=BATCH_SIZE,
)
history={
    'train_loss':[],
    'train_acc':[],
    'val_loss':[],
    'val_acc':[]
}
# 模型參數設定
model     = TwoStreamModel(num_classes=2)
loss_fn   = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)

# 訓練
best_val_loss =sys.float_info.max

for epoch in range(EPOCHS):

    # Train loop
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for spatial_input, temporal_input, labels in train_loader:
        spatial_input  = spatial_input.to(device)
        temporal_input = temporal_input.to(device)
        labels         = labels.to(device)

        predict = model(spatial_input, temporal_input)
        loss    = loss_fn(predict, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss    += loss.item()
        _, predicted   = torch.max(predict, 1)
        train_total   += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_loss /= len(train_loader)
    train_acc   = train_correct / train_total

    # Val loop
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for spatial_input, temporal_input, labels in val_loader:
            spatial_input  = spatial_input.to(device)
            temporal_input = temporal_input.to(device)
            labels         = labels.to(device)

            predict  = model(spatial_input, temporal_input)
            loss     = loss_fn(predict, labels)

            val_loss    += loss.item()
            _, predicted = torch.max(predict, 1)
            val_total   += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc   = val_correct / val_total

    print(f"Epoch {epoch+1:3d} | "
          f"Train  loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
          f"Val    loss: {val_loss:.4f}  acc: {val_acc:.4f}")
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    #每10個epoch存一次權重
    if (epoch +1)%10==0:
        torch.save(model.state_dict(),CKPT_DIR/ f"epoch_{epoch+1:02d}.pth")
        print(f"           → 儲存權重 (epoch {epoch+1:02d}) ")
    # 儲存最佳權重
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), CKPT_DIR / 'best_model.pth')
        print(f"           → 儲存最佳權重 (val_loss={best_val_loss:.4f})")
# --- 繪製並儲存 Loss 圖表 ---
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(history['train_loss']) + 1), history['train_loss'], label='Train Loss', color='blue')
plt.plot(range(1, len(history['val_loss']) + 1), history['val_loss'], label='Val Loss', color='red')
plt.title('Model Loss Progress')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(CKPT_DIR / 'loss_history.png')
plt.close()  # 關閉視窗，釋放記憶體以準備畫下一張
print(f"Loss 圖表已儲存至: {CKPT_DIR / 'loss_history.png'}")

# --- 繪製並儲存 Accuracy 圖表 ---
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(history['train_acc']) + 1), history['train_acc'], label='Train Acc', color='green')
plt.plot(range(1, len(history['val_acc']) + 1), history['val_acc'], label='Val Acc', color='orange')
plt.title('Model Accuracy Progress')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(CKPT_DIR / 'accuracy_history.png')
plt.close()
print(f"Accuracy 圖表已儲存至: {CKPT_DIR / 'accuracy_history.png'}")
print("\n--- 開始最終評估 (使用最佳模型) ---")
model.load_state_dict(torch.load(CKPT_DIR / 'best_model.pth'))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for spatial_input, temporal_input, labels in val_loader:
        spatial_input  = spatial_input.to(device)
        temporal_input = temporal_input.to(device)
        
        outputs = model(spatial_input, temporal_input)
        _, predicted = torch.max(outputs, 1)
        
        # 收集預測結果與真實標籤 (轉回 CPU 上的 numpy 格式)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# --- 1. 計算並儲存 F1-score 與 分類報告 ---
report = classification_report(all_labels, all_preds)
print("Classification Report:")
print(report)

# 將文字報告儲存為 txt 檔
with open(CKPT_DIR / 'evaluation_report.txt', 'w') as f:
    f.write(report)

# --- 2. 繪製並儲存混淆矩陣 ---
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[f'Class {i}' for i in range(2)], 
            yticklabels=[f'Class {i}' for i in range(2)])
plt.title('Confusion Matrix (Best Model)')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig(CKPT_DIR / 'confusion_matrix.png')
plt.close()

print(f"詳細報告已儲存至: {CKPT_DIR / 'evaluation_report.txt'}")
print(f"混淆矩陣圖已儲存至: {CKPT_DIR / 'confusion_matrix.png'}")