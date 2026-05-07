FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    apt-get update && apt-get install -y ffmpeg libx264-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /bdm

COPY  requirement.txt .

RUN pip install -r requirement.txt

COPY . .

EXPOSE 9999
EXPOSE 8000
EXPOSE 8501


CMD ["/bin/bash"]