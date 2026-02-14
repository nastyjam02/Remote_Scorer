import json
import random
import os
import argparse

def split_dataset(input_file, train_ratio=0.8, seed=None):
    """
    读取JSON数据集，随机打乱并按比例划分训练集和测试集。
    """
    if not os.path.exists(input_file):
        print(f"错误：文件 '{input_file}' 不存在。")
        return

    try:
        print(f"正在读取文件: {input_file} ...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print("错误：JSON文件内容不是列表格式，无法拆分。")
            return

        total_count = len(data)
        print(f"共读取到 {total_count} 条数据。")

        # 设置随机种子以保证可复现性（可选）
        if seed is not None:
            random.seed(seed)
            print(f"使用随机种子: {seed}")
        
        # 随机打乱
        print("正在随机打乱数据...")
        random.shuffle(data)

        # 计算切分点
        split_index = int(total_count * train_ratio)
        
        train_data = data[:split_index]
        test_data = data[split_index:]

        # 生成输出文件名
        dir_name = os.path.dirname(input_file)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        train_output_path = os.path.join(dir_name, f"{base_name}_train.json")
        test_output_path = os.path.join(dir_name, f"{base_name}_test.json")

        # 保存训练集
        print(f"正在保存训练集 ({len(train_data)} 条) 到: {train_output_path}")
        with open(train_output_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=4)

        # 保存测试集
        print(f"正在保存测试集 ({len(test_data)} 条) 到: {test_output_path}")
        with open(test_output_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=4)

        print("完成！")

    except json.JSONDecodeError:
        print(f"错误：文件 '{input_file}' 不是有效的JSON格式。")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将JSON数据集按比例随机拆分为训练集和测试集。")
    parser.add_argument("input_file", help="输入JSON文件的路径")
    parser.add_argument("--ratio", type=float, default=0.8, help="训练集比例 (默认: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认: 42，设为 -1 则不固定)")

    args = parser.parse_args()
    
    seed = args.seed if args.seed != -1 else None
    split_dataset(args.input_file, args.ratio, seed)
