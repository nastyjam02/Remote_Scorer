'''
使用预训练的 CLIP 或 SigLIP 模型对图像-文本对进行批量评分
'''
import argparse
import json
import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, SiglipProcessor, SiglipModel
from torch.utils.data import Dataset, DataLoader

class CaptionDataset(Dataset):
    def __init__(self, data, image_root):
        self.data = data
        self.image_root = image_root
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 兼容不同的键名
        if 'image_path' in item:
            img_path = item['image_path']
        elif 'filename' in item:
            img_path = item['filename']
        else:
            # Fallback
            img_path = ""
            
        if 'description' in item:
            text = item['description']
        elif 'caption' in item:
            text = item['caption']
        elif 'text' in item:
            text = item['text']
        else:
            text = ""
            
        basename = os.path.basename(img_path)
        full_image_path = os.path.join(self.image_root, basename)
        
        try:
            image = Image.open(full_image_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {full_image_path}: {e}")


            
        return {
            "image": image,
            "text": text,
            "original_item": item
        }

def collate_fn(batch):
    return {
        "images": [b['image'] for b in batch],
        "texts": [b['text'] for b in batch],
        "original_items": [b['original_item'] for b in batch]
    }

def main():
    parser = argparse.ArgumentParser(description="Batch score captions using CLIP/SigLIP")
    parser.add_argument("--json_input", required=True, help="Input JSON file")
    parser.add_argument("--json_output", required=True, help="Output JSON file with scores")
    parser.add_argument("--image_root", help="Root directory for images")
    parser.add_argument("--model_type", default= 'clip', choices=['clip', 'siglip'], help="Model type: clip or siglip")
    parser.add_argument("--model_name", type=str, default=None, help="HuggingFace model name (optional, uses defaults if not set)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Model
    if args.model_type == 'clip':
        model_name = args.model_name if args.model_name else "openai/clip-vit-base-patch32"
        print(f"Loading CLIP model: {model_name}")
        model = CLIPModel.from_pretrained(model_name).to(device)
        processor = CLIPProcessor.from_pretrained(model_name)
    else: # siglip
        model_name = args.model_name if args.model_name else "google/siglip-base-patch16-224"
        print(f"Loading SigLIP model: {model_name}")
        model = SiglipModel.from_pretrained(model_name).to(device)
        processor = SiglipProcessor.from_pretrained(model_name)
        
    model.eval()
    
    # Load Data
    with open(args.json_input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items")
    
    dataset = CaptionDataset(data, args.image_root)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    results = []
    
    for batch in tqdm(dataloader):
        images = batch['images']
        texts = batch['texts']
        original_items = batch['original_items']
        
        # Process inputs
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)

            logits_per_image = outputs.logits_per_image
            
            scores = logits_per_image.diag()
            

        scores_list = scores.cpu().tolist()
        
        for i, score in enumerate(scores_list):
            item = original_items[i]
            # Add score to the item
            if args.model_type == 'clip':
                item['clip_score'] = score
            else:
                item['siglip_score'] = score
            results.append(item)
            
    # Save results
    print(f"Saving results to {args.json_output}")
    with open(args.json_output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
if __name__ == "__main__":
    main()
