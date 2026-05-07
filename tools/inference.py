import torch
import sys
import cv2
import numpy as np
import tempfile
import subprocess
import os
from pathlib import Path
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.model import TwoStreamModel

CHECKPOINT_PATH = Path(__file__).parent.parent / 'checkpoints' / 'best_model.pth'
CLASS_NAMES = ['start', 'stop']
WINDOW_SIZE = 11
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_SPATIAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_model(checkpoint_path=CHECKPOINT_PATH, num_classes=2):
    model = TwoStreamModel(num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

def _compute_optical_flow(frames):
    h, w = frames[0].shape[:2]
    L = len(frames) - 1
    stacked = np.zeros((h, w, 2 * L), dtype=np.uint8)
    grid_u, grid_v = np.meshgrid(np.arange(w), np.arange(h))
    p_u, p_v = grid_u.astype(np.float32), grid_v.astype(np.float32)

    for k in range(L):
        prev_gray = cv2.cvtColor(frames[k], cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(frames[k + 1], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        dx=cv2.remap(flow[...,0],p_u,p_v,cv2.INTER_LINEAR)
        dy=cv2.remap(flow[...,1],p_u,p_v,cv2.INTER_LINEAR)
        dx -=np.mean(dx)
        dy -=np.mean(dy)
        p_u = np.clip(p_u + dx, 0, w - 1)
        p_v = np.clip(p_v + dy, 0, h - 1)
        stacked[..., 2 * k] = np.uint8(np.clip((dx + 20) * (255 / 40), 0, 255))
        stacked[..., 2 * k + 1] = np.uint8(np.clip((dy + 20) * (255 / 40), 0, 255))
    return stacked

def _preprocess_spatial(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return _SPATIAL_TRANSFORM(rgb).unsqueeze(0)

def _preprocess_temporal(flow):
    nc = flow.shape[2]
    resized = np.stack([cv2.resize(flow[..., c], (224, 224), interpolation=cv2.INTER_AREA) for c in range(nc)], axis=0)
    return torch.from_numpy(resized).float().unsqueeze(0) / 255.0

def run_inference(video_path, checkpoint_path=CHECKPOINT_PATH, on_window=None, infer_batch=32):
    model = load_model(checkpoint_path)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()

    if len(frames) < WINDOW_SIZE:
        raise ValueError(f"影片幀數不足，至少需要 {WINDOW_SIZE} 幀")

    num_windows = len(frames) // WINDOW_SIZE
    window_predictions = []

    with torch.no_grad():
        for batch_start in range(0, num_windows, infer_batch):
            batch_end = min(batch_start + infer_batch, num_windows)
            sp_list, tp_list = [], []
            for i in range(batch_start, batch_end):
                window = frames[i * WINDOW_SIZE:(i + 1) * WINDOW_SIZE]
                sp_list.append(_preprocess_spatial(window[WINDOW_SIZE // 2]))
                tp_list.append(_preprocess_temporal(_compute_optical_flow(window)))

            sp, tp = torch.cat(sp_list).to(DEVICE), torch.cat(tp_list).to(DEVICE)
            probs = torch.softmax(model(sp, tp), dim=1)

            for j in range(batch_end - batch_start):
                idx = probs[j].argmax().item()
                cls, conf = CLASS_NAMES[idx], float(probs[j, idx])
                window_predictions.append((cls, conf))
                if on_window: on_window(batch_start + j + 1, num_windows, cls, conf)

    votes = {}
    for cls, _ in window_predictions: votes[cls] = votes.get(cls, 0) + 1
    final_pred = max(votes, key=votes.get)
    final_conf = float(np.mean([c for cls, c in window_predictions if cls == final_pred]))

    # --- 修正影片寫入部分 ---
    h, w = frames[0].shape[:2]
    fd, raw_tmp_path = tempfile.mkstemp(suffix='.mp4')
    os.close(fd)

    # 嘗試多種編碼器，解決 Encoder not found
    codecs = ['mp4v', 'avc1', 'XVID']
    out = None
    for c in codecs:
        fourcc = cv2.VideoWriter_fourcc(*c)
        out = cv2.VideoWriter(raw_tmp_path, fourcc, fps, (w, h))
        if out.isOpened(): break

    if not out or not out.isOpened():
        raise RuntimeError("系統找不到可用的影片編碼器 (請安裝 ffmpeg)")

    for i, frame in enumerate(frames):
        win_idx = min(i // WINDOW_SIZE, num_windows - 1)
        cls, conf = window_predictions[win_idx]
        annotated = frame.copy()
        cv2.rectangle(annotated, (0, 0), (w, 55), (0, 0, 0), -1)
        cv2.putText(annotated, f"{cls} {conf:.1%}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        out.write(annotated)
    out.release()

    # --- 強制轉換為瀏覽器支援的 H.264 (Streamlit 播放關鍵) ---
    final_path = raw_tmp_path.replace(".mp4", "_web.mp4")
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', raw_tmp_path, 
            '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-loglevel', 'error',
            final_path
        ], check=True)
        os.remove(raw_tmp_path)
        return final_path, window_predictions, final_pred, final_conf
    except:
        # 如果 ffmpeg 轉檔失敗，就回傳原始路徑
        return raw_tmp_path, window_predictions, final_pred, final_conf