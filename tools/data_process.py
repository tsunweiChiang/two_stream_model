import cv2
import numpy as np
import os
from pathlib import Path

data_path = Path('/project/Research/two_stream_model/dataset/data')
spatial_path = Path('/project/Research/two_stream_model/dataset/spatial')
temporal_path = Path('/project/Research/two_stream_model/dataset/temporal')

# 設定參數
window_size = 11 

for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.lower().endswith('.mp4'):
            video_path = os.path.join(root, file)
            class_name = os.path.basename(root)
            
            print(f'\n' + '='*50)
            print(f'🚀 開始處理影片: {file}')
            print(f'📂 類別: {class_name}')

            spatial_output_path = spatial_path / class_name
            temporal_output_path = temporal_path / class_name
            spatial_output_path.mkdir(parents=True, exist_ok=True)
            temporal_output_path.mkdir(parents=True, exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            frame_count = len(frames)
            base_name = os.path.splitext(file)[0]

            if frame_count < window_size:
                print(f"⚠️ 警告: 影片幀數 ({frame_count}) 小於視窗大小 ({window_size})，已跳過。")
                continue

            print(f"🎬 總幀數: {frame_count} | 將產生 {frame_count // window_size} 個視窗")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            h, w = frames[0].shape[:2]

            # 開始滑動視窗處理
            for i in range(frame_count // window_size):
                # --- 1. 空間流處理 ---
                spatial_save_name = os.path.join(spatial_output_path, f"{base_name}_win_{i}.mp4")
                out = cv2.VideoWriter(spatial_save_name, fourcc, fps, (w, h))
                
                window_frames = frames[i * window_size : (i + 1) * window_size]
                for f in window_frames:
                    out.write(f)
                out.release()

                # --- 2. 時光流處理 ---
                L_flow = window_size - 1
                stacked_flow = np.zeros((h, w, 2 * L_flow), dtype=np.uint8)

                # 記錄一下光流的平均值，確認 Mean Subtraction 有運作
                flow_mean_list = []
                grid_u,grid_v=np.meshgrid(np.arange(w),np.arange(h))
                p_u=grid_u.astype(np.float32)
                p_v=grid_v.astype(np.float32)
                for k in range(L_flow):
                    prev_gray = cv2.cvtColor(window_frames[k], cv2.COLOR_BGR2GRAY)
                    next_gray = cv2.cvtColor(window_frames[k+1], cv2.COLOR_BGR2GRAY)

                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, next_gray, None, 
                        0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    

                    dx = cv2.remap(flow[...,0],p_u,p_v,cv2.INTER_LINEAR)
                    dy = cv2.remap(flow[...,1],p_u,p_v,cv2.INTER_LINEAR)

                    # 減去均值前先記錄原始平均位移 (用於 Print 觀察)
                    raw_mean = np.mean(np.sqrt(dx**2 + dy**2))
                    flow_mean_list.append(raw_mean)

                    # 平均流減法
                    dx -= np.mean(dx)
                    dy -= np.mean(dy)

                    p_u+=dx
                    p_v+=dy

                    p_u = np.clip(p_u, 0, w - 1)
                    p_v = np.clip(p_v, 0, h - 1)
                    # 正規化到 0-255
                    dx = np.uint8(np.clip((dx + 20) * (255 / 40), 0, 255))
                    dy = np.uint8(np.clip((dy + 20) * (255 / 40), 0, 255))

                    stacked_flow[..., 2 * k] = dx
                    stacked_flow[..., 2 * k + 1] = dy

                # 儲存時光流
                temporal_save_name = temporal_output_path / f"{base_name}_win_{i}_flow.npy"
                np.save(str(temporal_save_name), stacked_flow)

                # 每處理 5 個視窗或最後一個視窗時印一次進度
                if i % 5 == 0 or i == (frame_count - window_size):
                    avg_m = np.mean(flow_mean_list)
                    print(f"   📦 進度: [{i+1}/{frame_count - window_size + 1}] | 視窗平均位移: {avg_m:.4f} px")

            print(f"✅ 完成影片: {base_name}")
            print(f"   📂 Spatial 儲存至: {spatial_output_path}")
            print(f"   📂 Temporal 儲存至: {temporal_output_path}")

print("\n🎉 所有影片處理完畢！")