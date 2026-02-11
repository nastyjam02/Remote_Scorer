import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import dashscope
from dashscope import MultiModalConversation
from http import HTTPStatus

# ================= 配置区域 =================
# 请在这里设置您的 DashScope API Key
# 或者在环境变量中设置 DASHSCOPE_API_KEY
dashscope.api_key = "sk-ff7d1434f5e543eaa1bae0392909e5f8"

# 模型名称，根据您的需求可以改为 qwen-vl-plus, qwen-vl-max 或 qwen3-vl-plus (如果已发布)
MODEL_NAME = "qwen3-vl-plus"   

# 图像目录和输出路径
IMAGE_DIR = r"C:\Users\chenrui5-jk\Desktop\Score\UCM_imgs"
OUTPUT_JSON = r"C:\Users\chenrui5-jk\Desktop\Score\三元组数据-2000\qwen3vlplus\image_descriptions_t_07_2000_2.json"

# 提示词
PROMPT = (
    "Please describe the content of this remote sensing image in English. "
    "Provide a smooth and coherent description that includes the main objects, their quantities, and colors. "
    "Describe the spatial relationships between major objects and the background. "
    "Do not make inferences or assumptions beyond the facts. "
    "The description should be concise (within 150 words). "
    "\n\nReference Style Example:\n"
    "This is a high-resolution satellite image showing a large expanse of grassland with several trees scattered throughout. "
    "In the center of the image, there is a road running east to west, and in the top left and top right corners, "
    "there are two north-south roads intersecting it. There are several cars parked by the roadside. "
    "In the top of the image, there are several buildings with white roofs, and scattered planes and vehicles "
    "are parked on the grass around them."
)

# 并发数（根据 API 限制调整）
MAX_WORKERS = 5
# ===========================================

def get_caption(image_path):
    """调用 Qwen-VL API 获取图像描述"""
    # 构建输入消息格式
    # 注意：本地文件路径需要以 'file://' 开头
    image_url = f"file://{image_path}"
    
    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_url},
                {"text": PROMPT}
            ]
        }
    ]

    # 增加重试机制
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 加入 temperature 和 top_p 参数以增加生成的多样性
            response = MultiModalConversation.call(
                model=MODEL_NAME, 
                messages=messages,
                temperature=0.7,  # 增加随机性，避免每次生成完全一致的内容
                top_p=0.9         # 核采样，保证生成质量的同时保留多样性
            )
            
            if response.status_code == HTTPStatus.OK:
                # 提取生成的文本内容
                if response.output and response.output.choices and response.output.choices[0].message.content:
                    content = response.output.choices[0].message.content
                    # 某些情况下 content 可能是列表或对象
                    if isinstance(content, list):
                        for item in content:
                            if 'text' in item:
                                return item['text']
                    return str(content)
                return None
            elif response.code == 'Throttling.RateQuota':
                time.sleep(2 * (attempt + 1))
                continue
            else:
                print(f"Error: {image_path} - Code: {response.code}, Message: {response.message}")
                return None
        except Exception as e:
                print(f"Exception for {image_path} (Attempt {attempt+1}): {str(e)}")
                time.sleep(1)
    return None

def process_images():
    # 检查 API Key
    if dashscope.api_key == "YOUR_API_KEY_HERE" and not os.getenv("DASHSCOPE_API_KEY"):
        print("请在脚本中设置 dashscope.api_key 或设置环境变量 DASHSCOPE_API_KEY")
        return

    # 获取所有图片文件
    if not os.path.exists(IMAGE_DIR):
        print(f"错误: 找不到图像目录 {IMAGE_DIR}")
        return

    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(valid_extensions)]
    
    # 加载已有的 JSON 数据 (断点续传)
    results = []
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    results = json.loads(content)
            print(f"已加载 {len(results)} 条现有记录。")
        except Exception as e:
             print(f"读取现有文件失败: {e}。将从头开始或追加。")
    
    # 过滤已处理的图片
    processed_filenames = {item['filename'] for item in results}
    to_process = [f for f in image_files if f not in processed_filenames]
    
    print(f"总图像数: {len(image_files)}, 已处理: {len(processed_filenames)}, 待处理: {len(to_process)}")
    
    if not to_process:
        print("所有图像均已处理完毕。")
        return

    print("准备开始推理...")

    # 使用线程池进行批量处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 映射任务
        future_to_file = {executor.submit(get_caption, os.path.join(IMAGE_DIR, f)): f for f in to_process}
        
        count = 0
        total_new = len(to_process)
        
        # 使用 as_completed 实时获取结果
        for future in as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                caption = future.result()
                if caption:
                    results.append({
                        "filename": filename,
                        "caption": caption
                    })
                    count += 1
                    print(f"[{count}/{total_new}] 已完成: {filename}")
                    
                    # 每完成 5 张保存一次
                    if count % 50 == 0:
                        try:
                            with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                                json.dump(results, f, indent=4, ensure_ascii=False)
                            print(f"--- 进度已保存 ({len(results)}/{len(image_files)}) ---")
                        except Exception as save_err:
                            print(f"保存文件时出错: {save_err}")
                
                # 适当延时
                time.sleep(0.2)
            except Exception as e:
                print(f"处理 {filename} 时发生错误: {e}")

    # 最后保存一次
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\n全部推理完成！结果已保存至: {OUTPUT_JSON}")

if __name__ == "__main__":
    process_images()
