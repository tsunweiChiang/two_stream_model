import streamlit as st
import tempfile
import os
import sys
from pathlib import Path

# 確保能匯入 inference
sys.path.insert(0, str(Path(__file__).parent))
from inference import run_inference

st.set_page_config(page_title="Two-Stream Model Inference", layout="centered")
st.title("Two-Stream Model Inference")
st.caption("上傳影片，模型將每 11 幀做一次推論，並回傳標注結果影片。")

uploaded_file = st.file_uploader("請選擇要上傳的影片檔案", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.success(f"已成功上傳：{uploaded_file.name}")

    # 建立臨時檔案存儲上傳的影片
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()

    st.subheader("原始影片預覽")
    st.video(tfile.name)

    if st.button("開始推論", type="primary"):
        progress_bar = st.progress(0, text="初始化模型...")
        status_text = st.empty()
        window_rows = []

        def on_window(completed, total, cls, conf):
            pct = completed / total
            progress_bar.progress(pct, text=f"推論中：window {completed} / {total}")
            window_rows.append({"視窗": completed, "預測": cls, "信心度 (%)": round(conf * 100, 1)})

        try:
            # 執行推論
            out_path, preds, final_pred, final_conf = run_inference(
                tfile.name, on_window=on_window
            )
            
            progress_bar.progress(1.0, text="推論完成！")
            status_text.empty()

            st.success("推論完成！")
            st.subheader("推論結果影片")
            st.video(out_path)

            with open(out_path, 'rb') as f:
                st.download_button(
                    label="下載推論結果影片",
                    data=f,
                    file_name="inference_result.mp4",
                    mime="video/mp4",
                )

            st.subheader("各視窗預測明細")
            # 修正後的 Streamlit 語法
            st.dataframe(window_rows, width="stretch")

        except Exception as e:
            st.error(f"推論失敗：{e}")
        finally:
            # 清理臨時檔案
            if os.path.exists(tfile.name):
                os.unlink(tfile.name)

else:
    st.info("請上傳影片檔案以開始作業。")