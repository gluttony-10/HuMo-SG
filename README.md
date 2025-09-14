# HuMo-SG
原项目[HuMo](https://github.com/Phantom-video/HuMo)。针对单显卡进行了特化，并进行了破坏性修改，所以单独拉取了一个项目。顺手写了一个gradio界面。

## 开始安装
```sh
git clone https://github.com/Gluttony10/HuMo-SG.git
cd HuMo-SG
conda create -n humo python=3.10
conda activate humo
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install flash_attn
pip install -r requirements.txt
conda install -c conda-forge ffmpeg
```
torch版本支持50系显卡

## 下载模型
使用modelscope下载模型：
```sh
modelscope download --model Gluttony10/HuMo-SG --local_dir ./weights
```
对模型进行了量化和加速处理，节约了加载时间和推理时间，默认参数显存占用30G。

## 开始运行
直接运行：
```sh
python glut.py
```
局域网访问并修改端口：
```sh
python glut.py --server_name 0.0.0.0 --server_port 7861
```

## 参考项目
https://github.com/Phantom-video/HuMo