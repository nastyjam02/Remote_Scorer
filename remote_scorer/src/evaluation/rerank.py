import argparse
import json
import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, SiglipProcessor, SiglipModel

def main():
    parser = argparse.ArgumentParser(description="Rerank captions using CLIP/SigLIP")
    parser.add_argument("--input_json", required=True, help="Input JSON file with candidates")
    parser.add_argument("--output_json", required=True, help="Output JSON file with best caption")
    parser.add_argument("--image_root", default="", help="Root directory for images (optional if input_json contains absolute paths)")
    parser.add_argument("--model_type", default='siglip', choices=['clip', 'siglip'], help="Model type: clip or siglip")
    parser.add_argument("--model_name", type=str, default=None, help="HuggingFace model name")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Model
    if args.model_type == 'clip':
        model_name = args.model_name if args.model_name else "openai/clip-vit-large-patch14"  # clip-vit-large-patch14
        print(f"Loading CLIP model: {model_name}")
        model = CLIPModel.from_pretrained(model_name).to(device)
        processor = CLIPProcessor.from_pretrained(model_name)
    else: # siglip
        model_name = args.model_name if args.model_name else "google/siglip-base-patch16-224"   # siglip-base-patch16-224
        print(f"Loading SigLIP model: {model_name}")
        model = SiglipModel.from_pretrained(model_name).to(device)
        processor = SiglipProcessor.from_pretrained(model_name)
        
    model.eval()
    
    # Load Data
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items")
    
    results = []
    
    for item in tqdm(data):
        image_filename = item.get('image_path')
        candidates = item.get('candidates')
        
        if not image_filename or not candidates:
            continue
            
        # Load image
        # Determine image path: if image_root is provided, join with basename; otherwise use image_filename directly
        if args.image_root:
             image_path = os.path.join(args.image_root, os.path.basename(image_filename))
        else:
             # Assume image_filename is absolute or relative to current working directory
             image_path = image_filename
             # But if image_filename is just a basename like "1.tif", and no image_root is given, 
             # we might need a default assumption or just try opening it.
             # User instruction implies "1.tif" might be in the JSON, but they said 
             # "read path name from image_path" directly. 
             # If the JSON has relative paths like "1.tif" and no root is given, this will only work if CWD is correct.
             # However, typically "image_path" in these JSONs might already be absolute or relative to a known location.
             # Let's trust the user's instruction that the JSON contains the path info.
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            continue
            
        # Prepare inputs
        # CLIP/SigLIP expects a list of texts and one image (or list of images)
        # We want to compare 1 image against N texts
        
        try:
            inputs = processor(text=candidates, images=image, return_tensors="pt", padding=True, truncation=True).to(device)
        except Exception as e:
            print(f"Error processing inputs for {image_filename}: {e}")
            continue

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            
            if args.model_type == 'clip':
                # logits_per_image: [1, num_candidates]
                logits = outputs.logits_per_image
            else:
                # SigLIP
                logits = outputs.logits_per_image
                
        # Get best candidate
        # logits shape is [1, N]
        probs = logits.softmax(dim=1)
        best_idx = logits.argmax().item()
        
        best_caption = candidates[best_idx]
        best_score = logits[0, best_idx].item()
        
        # Store result
        # We can store just the best one, or rank them all.
        # User asked to "screen out the highest score caption"
        results.append({
            "image_path": image_filename,
            "caption": best_caption,
            "score": best_score,
        })
        
    # Save results
    print(f"Saving results to {args.output_json}")
    
    # Ensure directory exists
    output_dir = os.path.dirname(args.output_json)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
if __name__ == "__main__":
    main()

'''
python rerank.py \
  --input_json "/data/oceanus_ctr/j-chenrui5-jk/Score/三元组数据-2000/merged_candidates_2000_no_dedup.json" \
  --output_json "/data/oceanus_ctr/j-chenrui5-jk/Score/三元组数据-2000/scoreB/siglip-base-patch16-224.json"
'''