import torch
import sys
import cv2
import numpy as np
import tempfile
import subprocess
import os
from pathlib import Path
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.model import TwoStreamModel
import gc

CHECKPOINT_PATH = Path(__file__).parent.parent / 'checkpoints' / 'best_model.pth'
CLASS_NAMES = ['start', 'stop']
WINDOW_SIZE = 11
SLIDE_STEP = 5
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

def _compute_optical_flow(window):
    # 1. 【加速關鍵】先 Resize 到 224x224，大幅減少計算量
    small_window = [cv2.resize(f, (224, 224), interpolation=cv2.INTER_AREA) for f in window]
    
    h, w = 224, 224
    L = len(small_window) - 1
    stacked = np.zeros((h, w, 2 * L), dtype=np.uint8)
    
    # 建立網格，用於後續的 remap
    grid_u, grid_v = np.meshgrid(np.arange(w), np.arange(h))
    p_u, p_v = grid_u.astype(np.float32), grid_v.astype(np.float32)

    # 建議使用 DIS 光流，這在 CPU 上比 Farneback 快非常多(使用 fast)
    dis_flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

    for k in range(L):
        prev_gray = cv2.cvtColor(small_window[k], cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(small_window[k + 1], cv2.COLOR_BGR2GRAY)
        
        # 計算光流 (這裡會產生 dx 和 dy)
        flow = dis_flow.calc(prev_gray, next_gray, None)
        dx = flow[..., 0]
        dy = flow[..., 1]
        
        # 進行修正與運算 (現在 dx, dy 已經定義好了)
        # 如果你原本有使用 remap 邏輯：
        dx_remapped = cv2.remap(dx, p_u, p_v, cv2.INTER_LINEAR)
        dy_remapped = cv2.remap(dy, p_u, p_v, cv2.INTER_LINEAR)
        
        dx_remapped -= np.mean(dx_remapped)
        dy_remapped -= np.mean(dy_remapped)
        
        # 更新位置 (如果你的算法需要累積位移)
        p_u = np.clip(p_u + dx_remapped, 0, w - 1)
        p_v = np.clip(p_v + dy_remapped, 0, h - 1)

        # 映射到 0-255 並存入 stacked
        stacked[..., 2 * k] = np.uint8(np.clip((dx_remapped + 20) * (255 / 40), 0, 255))
        stacked[..., 2 * k + 1] = np.uint8(np.clip((dy_remapped + 20) * (255 / 40), 0, 255))
        
    return stacked

def _preprocess_spatial(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return _SPATIAL_TRANSFORM(rgb).unsqueeze(0)

def _preprocess_temporal(flow):
    # flow 形狀已經是 (224, 224, 2 * L)
    # 把通道 (Channel) 維度搬到最前面：(C, H, W)
    # 使用 transpose(2, 0, 1) 會比在裡面跑 cv2.resize 快非常多
    processed = flow.transpose(2, 0, 1) 
    return torch.from_numpy(processed).float().unsqueeze(0) / 255.0

def run_inference(video_path, checkpoint_path=CHECKPOINT_PATH, on_window=None, infer_batch=32):
    model = load_model(checkpoint_path)
    print(DEVICE)
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
    window_start=list(range(0, len(frames) - WINDOW_SIZE + 1, SLIDE_STEP))
    num_windows = len(window_start)
    window_predictions = []
    print(f"window_num:{num_windows}")
    with torch.no_grad():
        # 批次處理window
        for batch_start in range(0, num_windows, infer_batch):
            batch_indices = window_start[batch_start : batch_start + infer_batch]
            # 預處理每個 batch window 資料
            windows_in_batch = [frames[i : i + WINDOW_SIZE] for i in batch_indices]

            # 2. 處理空間流 (很快，維持原樣)
            sp_list = [_preprocess_spatial(win[WINDOW_SIZE // 2]) for win in windows_in_batch]

            # 3. 處理時間流 (最慢，使用並行加速)
            # 建立執行緒池，利用所有 CPU 核心
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                # 並行執行光流計算
                flow_results = list(executor.map(_compute_optical_flow, windows_in_batch))
        
            # 將計算好的光流張量化
            tp_list = [_preprocess_temporal(flow) for flow in flow_results]

            sp, tp = torch.cat(sp_list).to(DEVICE), torch.cat(tp_list).to(DEVICE)
            probs = torch.softmax(model(sp, tp), dim=1)

            # 回傳每個 window 的預測結果
            for j in range(len(batch_indices)):
                idx = probs[j].argmax().item()
                cls, conf = CLASS_NAMES[idx], float(probs[j, idx])
                window_predictions.append((cls, conf))
                if on_window: on_window(batch_start + j + 1, num_windows, cls, conf)
            
            del sp, tp, probs, sp_list, tp_list, flow_results
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
    except:
        final_path = raw_tmp_path
    
    del frames, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_path, window_predictions