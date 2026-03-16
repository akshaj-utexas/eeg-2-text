# import argparse
# import sys
# import os
# import torch
# from tqdm import tqdm
# from typing import List
# from collections import defaultdict
# import pandas as pd
# # Path injection for safety
# sys.path.append(os.getcwd())

# # from src.encoders import process_dream_diffusion
# from src.aligner import Aligner, calculate_noise
# from src.llm_client import LLMManager

# def parse_args():
#     parser = argparse.ArgumentParser(description="S2S: Neural Signal to Semantic Pipeline")
    
#     # Paths and Data
#     parser.add_argument("--dataset", type=str, default="imagenet_eeg_test", help="Path to the EEG dataset .pth file")
#     parser.add_argument("--word_corpus", type=str, default="imagenet", help="Path to the pre-encoded word corpus .pt file")
#     parser.add_argument("--output_dir", type=str, default="./results", help="Where to save embeddings and captions")
#     parser.add_argument("--batch_size", type=int, default=64, help="Batch size for EEG encoding")
    
#     # Model Config
#     parser.add_argument("--eeg_encoder", type=str, default="channelnet", help="Name of the EEG encoder model")
#     parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="cpu or cuda")
    
#     # Pipeline Logic
#     parser.add_argument("--llms", nargs="+", default=["openai"], help="LLMs to use (openai, google, anthropic)")
#     parser.add_argument("--top_k", type=int, default=15, help="Number of words to retrieve per EEG segment")
#     parser.add_argument("--sample_limit", type=int, default=None, help="Limit number of samples for LLM generation (to save credits)")
#     parser.add_argument("--skip_llm", action="store_true", help="Skip the LLM generation step")
#     parser.add_argument("--skip_eval", action="store_true", help="Skip the metrics calculation step")

#     return parser.parse_args()

# # import random 
# import random
# def main():
#     args = parse_args()
#     os.makedirs(args.output_dir, exist_ok=True)
    
#     print(f"--- Starting Baseline Pipeline with {args.eeg_encoder} ---")

#     # 1. & 2. EEG Encoding to CLIP Space
#     dataset_name = os.path.basename(args.dataset).replace('.pth', '')
#     clip_latents_path = "results/TESTING_channelnet_imagenet_eeg_test_clip_latents_with_pred_label_confidence.pt"
    
#     if not os.path.exists(clip_latents_path):
#         print("Step 1: Encoding EEG signals...")
#         if args.eeg_encoder == "channelnet":
#             from src.encoders import process_channelnet
#             process_channelnet(args.dataset, clip_latents_path, args.device, args.batch_size)
#     else:
#         print(f"Found existing latents at {clip_latents_path}, skipping encoding.")

#     # 3. Global Alignment (Baseline Approach)
#     print("Step 2: Performing Global Alignment...")

#     latent_dataset = torch.load(clip_latents_path)

#     # Load ONE global corpus for all subjects
#     global_corpus_path = "data/imagenet_corpus.pt" 
#     if not os.path.exists(global_corpus_path):
#         raise FileNotFoundError(f"Global corpus not found at {global_corpus_path}")

#     # Initialize a single global Aligner
#     global_aligner = Aligner(global_corpus_path, device=args.device)
    
#     aligned_results = []

#     # Perform Alignment across the entire dataset using the global corpus
#     for item in tqdm(latent_dataset, desc="Global Aligning"):
#         sub_id = item.get("subject")
#         pred_obj = item.get("predicted_object_label", None) 
#         pred_conf = item.get("prediction_confidence", 0.0) 
        
#         # Call Aligner using the Global Corpus (Noise centering removed)
#         bow_with_scores = global_aligner.align(
#             item['eeg_clip_latent'],
#             predicted_label=pred_obj,
#             confidence=pred_conf,
#             top_k=args.top_k
#         )
        
#         # Maintain your specific .pt structure for metrics compatibility
#         aligned_results.append({
#             "subject": sub_id,
#             "gt_object_label": item.get("object_label", ""),   
#             "gt_caption": item.get("caption", ""),            
#             "predicted_object_label": pred_obj,               
#             "prediction_confidence": pred_conf,                 
#             "bow": bow_with_scores,                           
#             "prompt_words": [w['word'] for w in bow_with_scores]
#         })

#     # Save results to the specified baseline path
#     aligned_path = os.path.join(args.output_dir, f"{dataset_name}_naive_baseline.pt")
#     torch.save(aligned_results, aligned_path)
#     print(f"✅ Step 2 Complete. Naive baseline results saved to {aligned_path}")
#     exit()
#     if not args.skip_llm:
#         print("Step 3: Generating Captions via LLM APIs...")
#         boosted_results_path = "results/imagenet_eeg_test_boosted_results_with_conf.pt"
#         latent_dataset = torch.load(boosted_results_path)
        

#         # 3. Proceed with LLM Generation
#         llm_manager = LLMManager(provider="openai", model_name="gpt-4o-mini-2024-07-18")
#         final_output_path = os.path.join(args.output_dir, "all_subjects_feb20_linear_boost.pt")

#         # Pass the sampled list directly to the experiment runner
#         final_generations = llm_manager.run_decoding_experiment(
#             input_path=boosted_results_path, # Update your method to accept this
#             output_path=final_output_path
#         )
#         print(f"✅ Pipeline Complete. Samples saved to {final_output_path.replace('.pt', '.csv')}")
#     # 5. Evaluation (To be implemented)
#     if not args.skip_eval:
#         print("Step 4: Running Evaluation Metrics (Coming Soon)...")

# if __name__ == "__main__":
#     main()

import argparse
import sys
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd

# Path injection
sys.path.append(os.getcwd())

from src.aligner import Aligner
from src.llm_client import LLMManager
from src.models import SimilarityRefiner
from src.trainer import Stage1_5Dataset, run_training
from src.metrics import evaluate_and_save_metrics
from src.encoders import DATASET_REGISTRY

def parse_args():

    parser = argparse.ArgumentParser(description="SENSE: SEmantic Neural Sparse Extraction Pipeline")
    # Paths and Data
    parser.add_argument("--dataset", type=str, default="imagenet_eeg_test", help="Path to the EEG dataset .pth file")
    parser.add_argument("--vocab_path", type=str, default="data/imagenet_train_corpus.pt", help="Path to the encoded word corpus")
    parser.add_argument("--output_dir", type=str, default="./pipeline_test", help="Where to save outputs")
    parser.add_argument("--batch_size", type=int, default=64)
    
    # Mode Selection
    parser.add_argument("--mode", type=str, choices=["naive", "train", "inference"], default="naive", 
                        help="naive: Cosine Sim only | train: Train MLP | inference: Use trained MLP")
    
    # MLP Config
    parser.add_argument("--loss", type=str, choices=["bce", "focal", "contrastive"], default="bce")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pth weights for inference")
    parser.add_argument("--eeg_encoder", type=str, default="channelnet")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # LLM & Eval Logic
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--skip_llm", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")

    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    dataset_name = os.path.basename(args.dataset).replace('.pth', '')
    
    # 1. EEG Encoding Step (Assumes raw EEG -> CLIP latents)
    # This logic checks for existing latents to save time/compute
    clip_latents_path = os.path.join(args.output_dir, f"{dataset_name}_pipeline_test_latents.pt")
    
    if not os.path.exists(clip_latents_path):
        print(f"--- Step 1: Encoding EEG via {args.eeg_encoder} ---")
        from src.encoders import process_channelnet
        process_channelnet(args.dataset, clip_latents_path, args.device, args.batch_size)
    else:
        print(f"--- Found existing latents at {clip_latents_path} ---")

    # 2. Alignment Logic (The Core Switch)
    final_alignment_path = ""

    dataset_path = DATASET_REGISTRY.get(args.dataset, None)
    print(f"Dataset path for training: {dataset_path}")

    if args.mode == "train":
        print(f"--- Mode: Training MLP ({args.loss} loss) ---")
        train_ds = Stage1_5Dataset(clip_latents_path, args.vocab_path)
        loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        
        # Initialize model: Contrastive loss uses 'NoScaling' (use_scaling=False)
        model = SimilarityRefiner(train_ds.vocab_embeddings, use_scaling=(args.loss != "contrastive"))
        
        save_name = f"mlp_{args.eeg_encoder}_{args.loss}_{args.epochs}eps.pth"
        save_path = os.path.join("checkpoints", save_name)
        
        run_training(model, loader, args.device, args.epochs, args.loss, save_path)
        print(f"Training complete. Model saved to {save_path}. Exiting.")
        return # Training usually stops here before inference

    elif args.mode == "inference":
        print(f"--- Mode: MLP Inference using {args.checkpoint} ---")
        if not args.checkpoint or not os.path.exists(args.checkpoint):
            raise ValueError("Inference mode requires a valid --checkpoint path.")

        vocab_info = torch.load(args.vocab_path)
        model = SimilarityRefiner(vocab_info["embeddings"])
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
        model.to(args.device).eval()
        
        latent_dataset = torch.load(clip_latents_path)
        aligned_results = []

        for item in tqdm(latent_dataset, desc="MLP Mapping"):
            eeg_vec = item['eeg_clip_latent'].to(args.device).float()
            if eeg_vec.dim() == 1: eeg_vec = eeg_vec.unsqueeze(0)

            with torch.no_grad():
                logits, refined_latent = model(eeg_vec)
                probs = torch.sigmoid(logits).squeeze()
            
            scores, indices = probs.topk(min(args.top_k, len(vocab_info["words"])))
            bow = [{"word": vocab_info["words"][idx], "score": s.item()} for s, idx in zip(scores, indices)]

            aligned_results.append({
                "subject": item.get("subject"),
                "gt_object_label": item.get("object_label", ""),
                "gt_caption": item.get("caption", ""),
                "predicted_object_label": item.get("predicted_object_label", "n/a"),
                "prediction_confidence": item.get("prediction_confidence", 0.0),
                "bow": bow,
                "prompt_words": [w['word'] for w in bow],
                "refined_latent": refined_latent.cpu()
            })
        
        final_alignment_path = os.path.join(args.output_dir, f"{dataset_name}_mlp_{args.loss}_aligned.pt")
        torch.save(aligned_results, final_alignment_path)

    elif args.mode == "naive":
        print("--- Mode: Naive Global Alignment ---")
        global_aligner = Aligner(args.vocab_path, device=args.device)
        latent_dataset = torch.load(clip_latents_path)
        aligned_results = []

        for item in tqdm(latent_dataset, desc="Naive Aligning"):
            bow = global_aligner.align(
                item['eeg_clip_latent'], 
                top_k=args.top_k
            )
            aligned_results.append({
                "subject": item.get("subject"),
                "gt_object_label": item.get("object_label", ""),
                "gt_caption": item.get("caption", ""),
                "predicted_object_label": item.get("predicted_object_label", "n/a"),
                "prediction_confidence": item.get("prediction_confidence", 0.0),
                "bow": bow,
                "prompt_words": [w['word'] for w in bow]
            })
        
        final_alignment_path = os.path.join(args.output_dir, f"{dataset_name}_naive_aligned.pt")
        torch.save(aligned_results, final_alignment_path)

    # 3. Semantic Decoding via LLM
    if not args.skip_llm and final_alignment_path:
        print(f"--- Step 3: LLM Caption Generation ---")
        llm_manager = LLMManager(provider="openai", model_name="gpt-4o-mini")
        
        final_gen_path = final_alignment_path.replace(".pt", "_captions.pt")
        llm_manager.run_decoding_experiment(
            input_path=final_alignment_path,
            output_path=final_gen_path
        )

    final_csv_path = "results/gemini/mlp_test_llm_gemini_focal_loss_learnt_scaling.csv"
    # 4. Evaluation
    if not args.skip_eval and final_csv_path:
        print(f"--- Step 4: Running Metrics ---")
        evaluate_and_save_metrics(final_csv_path, output_dir=args.output_dir)

if __name__ == "__main__":
    main()