import json
import os
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

class COCOEvalCap:
    def __init__(self, images, gts, res):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.params = {'image_id': images}
        self.gts = gts
        self.res = res

    def evaluate(self):
        imgIds = self.params['image_id']
        gts = self.gts
        res = self.res

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]

        # =================================================
        # Compute scores
        # =================================================
        eval = {}
        for scorer, method in scorers:
            print('computing %s score...' % (scorer.method()))
            try:
                score, scores = scorer.compute_score(gts, res)
            except Exception as e:
                print(f"Error computing score for {scorer.method()}: {e}")
                continue

            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f" % (m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f" % (method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [self.imgToEval[imgId] for imgId in self.params['image_id']]

def normalize_path(path):
    # Extract basename to use as ID (e.g., "1.tif")
    return os.path.basename(path)

def load_ground_truth(file_path):
    print(f"Loading Ground Truth from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    gts = {}
    for item in data:
        img_id = normalize_path(item['image_path'])
        caption = item['describe']
        # GT needs to be a list of captions
        if img_id not in gts:
            gts[img_id] = []
        gts[img_id].append({"caption": caption})
    return gts

def load_candidates(file_path):
    print(f"Loading Candidates from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    res = {}
    for item in data:
        img_id = normalize_path(item['image_path'])
        # Try different keys for caption
        caption = item.get('caption') or item.get('text') or item.get('describe')
        
        if caption:
            res[img_id] = [{"caption": caption}]
            
    return res

def main():
    gt_file = "clipscore/UCM_captions_2000.json"
    
    candidate_files = [
         "clipscore/best_scores_merged.json",
        "clipscore/clip-vit-base-patch32.json",
        "clipscore/clip-vit-large-patch14.json",
        "clipscore/siglip-base-patch16-224.json",
        "clipscore/siglip-so400m-patch14-384.json",
        "clipscore/random.json"
    ]
    
    # Load GT
    gts = load_ground_truth(gt_file)
    image_ids = list(gts.keys())
    print(f"Loaded {len(image_ids)} ground truth images.")

    results_summary = {}

    for cand_file in candidate_files:
        model_name = os.path.basename(cand_file).replace(".json", "")
        print(f"\nEvaluating {model_name}...")
        
        res = load_candidates(cand_file)
        
        # Filter: ensure we only evaluate images present in both GT and Res
        # (Though ideally they should match exactly)
        common_ids = [img_id for img_id in image_ids if img_id in res]
        print(f"Evaluated on {len(common_ids)} common images.")
        
        if len(common_ids) == 0:
            print("No common images found! Skipping.")
            continue
            
        # Subset dictionaries
        gts_subset = {k: gts[k] for k in common_ids}
        res_subset = {k: res[k] for k in common_ids}
        
        # Evaluate
        try:
            eval_cap = COCOEvalCap(common_ids, gts_subset, res_subset)
            eval_cap.evaluate()
            results_summary[model_name] = eval_cap.eval
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Print Summary Table
    print("\n" + "="*95)
    print(f"{'Model':<30} {'CIDEr':<10} {'Bleu_4':<10} {'METEOR':<10} {'ROUGE_L':<10}")
    print("-" * 95)
    for model, metrics in results_summary.items():
        print(f"{model:<30} {metrics.get('CIDEr', 0):<10.3f} {metrics.get('Bleu_4', 0):<10.3f} {metrics.get('METEOR', 0):<10.3f} {metrics.get('ROUGE_L', 0):<10.3f}")
    print("="*95)
    
    # Save detailed results to JSON
    output_json = "eval_ranking_results.json"
    try:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=4, ensure_ascii=False)
        print(f"\nDetailed evaluation results saved to: {output_json}")
    except Exception as e:
        print(f"Error saving results json: {e}")

if __name__ == "__main__":
    main()
