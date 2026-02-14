import os
import json
import argparse
import time
from openai import OpenAI
from typing import List, Dict

OPENAI_COMPATIBLE_BASE_URL = ""
MODEL_NAME = "qwen3-max" 

# 用于生成硬负样本的Prompt模板 (高级版)
# 将Prompt分为系统角色和用户角色
SYSTEM_PROMPT = """
你是一位顶级的AI数据集标注专家。你的任务是为给定的“正面样本”创建一个包含2-3处细节错误的硬负样本。

**# 任务执行流程 (必须严格遵守)**

你必须按照以下三步思考并输出：

**【第一步：分析】**
- 阅读用户提供的“正面样本”。
- 识别出句子中 **两到三个** 可以被修改的具体、可观察的细节（例如：颜色、数量、形状、位置、动作）。
- 在你的思考过程中，列出这些可以修改的点。

**【第二步：计划】**
- 针对你在第一步中找到的2-3个点，决定具体的修改方案。
- 例如：将“红色”改为“蓝色”，将“五”改为“六”，将“坐着”改为“站着”。

**【第三步：生成】**
- 根据第二步的计划，将所有修改应用到原始句子中，生成最终的硬负样本。
- **你的最终输出必须且只能是生成的句子本身**，格式如下：
`最终生成的句子：[这里是修改后的句子]`

---
**# 示例**

**用户输入:**
正面样本 (pos):

晴天下的公园里，一个穿着红色T恤的男孩正坐在长椅上，手里拿着一本蓝色的书在阅读。

**你的思考与输出:**
【第一步：分析】
原始句子中有多个可修改细节：
1.  T恤颜色：“红色”
2.  男孩的姿势：“坐着”
3.  书的颜色：“蓝色”

【第二步：计划】
我将修改其中两处：
1.  将T恤颜色从“红色”改为“绿色”。
2.  将姿势从“坐在长椅上”改为“站在长椅旁”。

【第三步：生成】
最终生成的句子：晴天下的公园里，一个穿着绿色T恤的男孩正站在长椅旁，手里拿着一本蓝色的书在阅读。
"""


# 初始化OpenAI客户端
# 它会自动从环境变量 `DASHSCOPE_API_KEY` 读取密钥
try:
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=OPENAI_COMPATIBLE_BASE_URL,
    )
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise ValueError("错误：环境变量 DASHSCOPE_API_KEY 未设置。")
except Exception as e:
    client = None
    print(f"初始化API客户端时出错: {e}")


# --- 函数定义 ---

def call_qwen_api(pos_text: str) -> str:
    """
    使用OpenAI兼容模式调用通义千问API生成文本。

    Args:
        pos_text: 正面样本的文本。

    Returns:
        模型生成的文本内容或错误信息。
    """
    if not client:
        return "API_CLIENT_INIT_ERROR: 客户端未成功初始化。"

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f'正面样本 (pos):\n"""\n{pos_text}\n"""'}
            ],
            temperature=0.3,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"API_CALL_ERROR: {e}"

def process_json_file(input_path: str, output_path: str, limit: int = None):
    """
    读取JSON文件，为每个条目生成hard_neg，并保存到新文件。
    """
    if not client:
        print("API客户端未初始化，无法处理文件。")
        return

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误：无法读取或解析输入文件 {input_path}。 {e}")
        return

    # 如果设置了limit，则只处理前N条
    if limit is not None and limit > 0:
        print(f"--- 测试模式：将只处理前 {limit} 个样本。 ---")
        data = data[:limit]

    results: List[Dict] = []
    total_items = len(data)

    print(f"共找到 {total_items} 个条目。开始处理...")

    for i, item in enumerate(data):
        pos_text = item.get("pos")
        
        if not pos_text:
            print(f"警告：第 {i+1}/{total_items} 个条目缺少 'pos' 字段，已跳过。")
            results.append(item)
            continue

        print(f"正在处理第 {i+1}/{total_items} 个条目...")
        raw_output = call_qwen_api(pos_text)
        
        # --- 新增：解析模型的输出，只提取最终句子 ---
        parsed_neg = raw_output
        marker = "最终生成的句子："
        if marker in raw_output:
            # 分割字符串并获取标记后的部分
            parsed_neg = raw_output.split(marker)[-1].strip()
        elif "【第三步：生成】" in raw_output:
            # 备用方案，如果模型忘了写“最终生成的句子：”
            parsed_neg = raw_output.split("【第三步：生成】")[-1].strip()

        new_item = item.copy()
        new_item["hard_neg"] = parsed_neg
        results.append(new_item)

        print(f"  - Pos: {pos_text[:50]}...")
        print(f"  - Hard Neg: {parsed_neg[:50]}...")
        
        time.sleep(1)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"\n处理完成！结果已保存到：{output_path}")
    except IOError as e:
        print(f"错误：无法写入输出文件 {output_path}。 {e}")

# --- 主程序入口 ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用通义千问模型为JSON文件生成'细节错误型'硬负样本。")
    parser.add_argument("input_file", help="输入的JSON文件路径。")
    parser.add_argument("output_file", help="输出的JSON文件路径。")
    parser.add_argument("--limit", type=int, default=None, help="要处理的样本数量上限，用于测试。")
    
    args = parser.parse_args()

    process_json_file(args.input_file, args.output_file, args.limit)