import sys

path_to_insert = "humo"
if path_to_insert not in sys.path:
    sys.path.insert(0, path_to_insert)

from common.config import load_config, create_object

import psutil
import argparse
import torch
import gradio as gr
import random
import numpy as np
import json
import os
import yaml

parser = argparse.ArgumentParser() 
parser.add_argument("--server_name", type=str, default="127.0.0.1", help="IP地址，局域网访问改为0.0.0.0")
parser.add_argument("--server_port", type=int, default=7891, help="使用端口")
parser.add_argument("--share", action="store_true", help="是否启用gradio共享")
parser.add_argument("--mcp_server", action="store_true", help="是否启用mcp服务")
args = parser.parse_args()

print(" 启动中，请耐心等待 bilibili@十字鱼 https://space.bilibili.com/893892")
print(f'\033[32mPytorch版本：{torch.__version__}\033[0m')
if torch.cuda.is_available():
    device = "cuda" 
    print(f'\033[32m显卡型号：{torch.cuda.get_device_name()}\033[0m')
    total_vram_in_gb = torch.cuda.get_device_properties(0).total_memory / 1073741824
    print(f'\033[32m显存大小：{total_vram_in_gb:.2f}GB\033[0m')
    mem = psutil.virtual_memory()
    print(f'\033[32m内存大小：{mem.total/1073741824:.2f}GB\033[0m')
    if torch.cuda.get_device_capability()[0] >= 8:
        print(f'\033[32m支持BF16\033[0m')
        dtype = torch.bfloat16
    else:
        print(f'\033[32m不支持BF16，仅支持FP16\033[0m')
        dtype = torch.float16
else:
    print(f'\033[32mCUDA不可用，请检查\033[0m')
    device = "cpu"


def generate(
    audio,
    prompt,
    negative_prompt,
    image,
    resolution,
    num_frames,
    seed
):
    try:
        if seed < 0:
            seed = random.randint(0, np.iinfo(np.int32).max)
        else:
            seed = seed

        if resolution == "1280*720":
            width = 1280
            height = 720
        else:
            width = 832
            height = 480

        data = {
            "glut": {
                "img_paths": [image] if image else "",  # 处理空图片情况
                "audio_path": audio,
                "prompt": prompt
            }
        }

        # 写入JSON文件（保留原有格式）
        json_path = "glut.json"
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            existing_data.update(data)
            data = existing_data
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        mode="TIA" if image else "TA"

        # 读取当前配置文件路径
        config_path = "glut.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 更新字段
        config['generation']['mode'] = mode
        config['generation']['width'] = width
        config['generation']['height'] = height
        config['generation']['frames'] = num_frames
        config['generation']['sample_neg_prompt'] = negative_prompt
        config['generation']['seed'] = seed

        # 写回文件
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, allow_unicode=True, default_flow_style=False)

        # Load config
        config = load_config(config_path)
        runner = create_object(config)
        runner.entrypoint()

        return f"outputs/glut_seed{seed}.mp4", f"种子数{seed}，保存在outputs/glut_seed{seed}.mp4"
    
    except Exception as e:
        error_msg = f"发生错误：{str(e)}"
        print(f'\033[31m{error_msg}\033[0m')  # 控制台红色显示
        return None, error_msg  # 返回空视频路径和错误信息

with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("""
            <div>
                <h2 style="font-size: 30px;text-align: center;">HuMo</h2>
            </div>
            <div style="text-align: center;">
                十字鱼
                <a href="https://space.bilibili.com/893892">🌐bilibili</a> 
                |HuMo
                <a href="https://github.com/Phantom-video/HuMo">🌐github</a> 
            </div>
            <div style="text-align: center; font-weight: bold; color: red;">
                ⚠️ 该演示仅供学术研究和体验使用。
            </div>
            """)

    with gr.TabItem("音频驱动"):
        with gr.Row():
            with gr.Column():
                audio = gr.Audio(label="输入音频", type="filepath")
                prompt = gr.Textbox(label="提示词", info="英文提示词效果更佳", value="远景，女人在唱歌")
                negative_prompt = gr.Textbox(label="负面提示词", value="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
                with gr.Accordion("输入图片（可选）", open=False):
                    image = gr.Image(label="输入图片", type="filepath", height=400)
                with gr.Row():
                    generate_button = gr.Button("🎬 开始生成", variant='primary')
                with gr.Accordion("参数设置", open=True):
                    resolution = gr.Dropdown(label="分辨率", choices=["1280*720", "832*480"], value="832*480")
                    num_frames = gr.Slider(label="总帧数", info="=秒数x25+1", minimum=26, maximum=2001, step=25, value=76)
                    seed = gr.Slider(label="种子", minimum=-1, maximum=2147483647, step=1, value=-1)
            with gr.Column():
                info = gr.Textbox(label="提示信息", interactive=False)
                video_output = gr.Video(label="生成结果", interactive=False)

    gr.on(
        triggers=[generate_button.click],
        fn = generate,
        inputs = [
            audio,
            prompt,
            negative_prompt,
            image,
            resolution,
            num_frames,
            seed
        ],
        outputs = [video_output, info]
    )


if __name__ == "__main__": 
    demo.launch(
        server_name=args.server_name, 
        server_port=args.server_port,
        share=args.share, 
        mcp_server=args.mcp_server,
        inbrowser=True,
    )
