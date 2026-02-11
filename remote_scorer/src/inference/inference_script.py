"""
推理脚本
"""
import argparse
import logging
import json
from tqdm import tqdm
from config import ModelConfig, DataConfig
from inference import SimilarityScorer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="为遥感图像和文本描述计算相似度分数")
    parser.add_argument("--checkpoint", type=str, required=True, help="训练好的模型检查点目录 (例如: output/training_.../best_model/)")
    parser.add_argument("--image_path", type=str, help="单张待评分图像的路径")
    parser.add_argument("--text", type=str, help="与单张图像匹配的文本描述")
    parser.add_argument("--json_input", type=str, help="包含图文对的JSON文件路径")
    parser.add_argument("--json_output", type=str, help="保存JSON评分结果的路径")
    args = parser.parse_args()

    # 初始化评分器 (不需要手动加载 ModelConfig，直接从 checkpoint 目录加载)
    logging.info("初始化评分器...")
    try:
        scorer = SimilarityScorer(args.checkpoint)
    except Exception as e:
        logging.error(f"评分器初始化失败: {e}")
        return

    if args.image_path and args.text:
        # 单张图像评分模式
        logging.info("单张图像评分...")
        score = scorer.score(args.image_path, args.text)
        level = scorer._score_to_level(score)
        
        print(f"图像: {args.image_path}")
        print(f"文本: {args.text}")
        print(f"相似度分数: {score:.4f}")
        print(f"质量等级: {level}")

    elif args.json_input:
        # JSON批量评分模式
        logging.info(f"从 {args.json_input} 进行批量评分...")
        try:
            with open(args.json_input, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logging.error(f"无法读取或解析JSON文件: {e}")
            return
            
        results = []
        # 使用 tqdm 显示进度
        iterator = tqdm(data, desc="推理进度", unit="样本")
        for item in iterator:
            image_path = item.get("image_path")
            text = item.get("text")
            if image_path and text:
                score = scorer.score(image_path, text)
                level = scorer._score_to_level(score)
                results.append({
                    "image_path": image_path,
                    "text": text,
                    "similarity_score": score,
                    "quality_level": level
                })
        
        if args.json_output:
            try:
                with open(args.json_output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                logging.info(f"评分结果已保存至: {args.json_output}")
            except Exception as e:
                logging.error(f"无法保存结果到JSON文件: {e}")
        else:
            # 如果未指定输出文件，则打印到控制台
            print(json.dumps(results, ensure_ascii=False, indent=4))

    else:
        logging.warning("请输入有效的参数。使用 --image_path 和 --text 进行单张评分，或使用 --json_input 进行批量评分。")

if __name__ == "__main__":
    main()
