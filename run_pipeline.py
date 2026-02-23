# import argparse
# import sys
# from typing import List
# import os
# from src.encoders import process_dream_diffusion
# from src.aligner import Aligner, calculate_noise
# from src.llm_client import LLMManager

# def parse_args():
#     parser = argparse.ArgumentParser()
    
#     # Paths and Data
#     parser.add_argument("--dataset", type=str, default="imagenet_eeg", help="Path to the EEG dataset")
#     # parser.add_argument("--word_corpus", type=str, required=True, help="Path to the word corpus file")
#     parser.add_argument("--output_dir", type=str, default="./results", help="Where to save embeddings and captions")
#     parser.add_argument("--batch_size", type=int, default=64)
    
#     # Model Config
#     parser.add_argument("--eeg_encoder", type=str, default="dream_diffusion", help="Name or path of the EEG encoder model")
#     parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    
#     # Pipeline Logic
#     # parser.add_argument("--llms", nargs="+", default=["gpt-4"], help="LLMs to use (gpt-4, gemini, claude)")
#     # parser.add_argument("--top_k", type=int, default=10, help="Number of words to retrieve per EEG segment")
#     # parser.add_argument("--skip_eval", action="store_true", help="Skip the metrics calculation step")

#     return parser.parse_args()

# def main():
#     args = parse_args()
#     os.makedirs(args.output_dir, exist_ok=True)
    
#     print(f"--- Starting Pipeline with {args.eeg_encoder} ---")

#     # 1. & 2. Load Data and Encode EEG -> CLIP Space
#     # We combine these because the encoder logic handles the loading/saving internally for now
#     clip_latents_path = os.path.join(args.output_dir, f"{args.eeg_encoder}_{args.dataset}_clip_latents.pt")
    
#     if args.eeg_encoder == "dream_diffusion":
#         process_dream_diffusion(args.dataset, clip_latents_path, args.device, args.batch_size)

#     # works till here ^^

#     # aligner = Aligner(args.word_corpus, device=args.device)
#     # dataset = torch.load(clip_latents_path)
#     # noise = calculate_noise(dataset, device=args.device)

#     # final_results = []
#     # for item in dataset:
#     #     bow_with_scores = aligner.align(item['eeg_clip_latent'], noise, top_k=args.top_k)
#     #     final_results.append({
#     #         "subject": item.get["subject"],
#     #         "gt_caption": item.get["caption"],
#     #         "bow": bow_with_scores  # contains words and their similarity scores as floats
#     #         "prompt_words": [w['word'] for w in bow_with_scores]  # Just the words for LLM input
#     #     })

#     # torch.save(final_results, os.path.join(args.output_dir, "aligned_results.pt"))
    
    
#     # 3. Retrieve Words from Corpus (Next step)
#     # 4. LLM Generation (Next step)
#     # 5. Evaluation (Next step)

# if __name__ == "__main__":
#     main()

import argparse
import sys
import os
import torch
from tqdm import tqdm
from typing import List
from collections import defaultdict
import pandas as pd
# Path injection for safety
sys.path.append(os.getcwd())

# from src.encoders import process_dream_diffusion
from src.aligner import Aligner, calculate_noise
from src.llm_client import LLMManager

def parse_args():
    parser = argparse.ArgumentParser(description="S2S: Neural Signal to Semantic Pipeline")
    
    # Paths and Data
    parser.add_argument("--dataset", type=str, default="imagenet_eeg_train", help="Path to the EEG dataset .pth file")
    parser.add_argument("--word_corpus", type=str, default="imagenet", help="Path to the pre-encoded word corpus .pt file")
    parser.add_argument("--output_dir", type=str, default="./results", help="Where to save embeddings and captions")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for EEG encoding")
    
    # Model Config
    parser.add_argument("--eeg_encoder", type=str, default="channelnet", help="Name of the EEG encoder model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="cpu or cuda")
    parser.add_argument("--boost_mode", type=str, default="linear", help="Boosting mode for predicted label (linear, additive, multiplicative)")
    
    # Pipeline Logic
    parser.add_argument("--llms", nargs="+", default=["openai"], help="LLMs to use (openai, google, anthropic)")
    parser.add_argument("--top_k", type=int, default=10, help="Number of words to retrieve per EEG segment")
    parser.add_argument("--sample_limit", type=int, default=None, help="Limit number of samples for LLM generation (to save credits)")
    parser.add_argument("--skip_llm", action="store_true", help="Skip the LLM generation step")
    parser.add_argument("--skip_eval", action="store_true", help="Skip the metrics calculation step")

    return parser.parse_args()

# import random 
import random
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"--- Starting Pipeline with {args.eeg_encoder} ---")

    # 1. & 2. EEG Encoding to CLIP Space
    # Using os.path.basename to create a clean filename for latents
    dataset_name = os.path.basename(args.dataset).replace('.pth', '')
    clip_latents_path = os.path.join(args.output_dir, f"{args.eeg_encoder}_{dataset_name}_train_clip_latents_with_pred_label_confidence.pt")
    
    # Check if latents already exist to save time/compute
    if not os.path.exists(clip_latents_path):
        print("Step 1: Encoding EEG signals...")
        if args.eeg_encoder == "dream_diffusion":
            process_dream_diffusion(args.dataset, clip_latents_path, args.device, args.batch_size)
        if args.eeg_encoder == "channelnet":
            from src.encoders import process_channelnet
            process_channelnet(args.dataset, clip_latents_path, args.device, args.batch_size)
    else:
        print(f"Found existing latents at {clip_latents_path}, skipping encoding.")
    exit()
    # 3. Retrieve Words from Corpus (Alignment)
    print("Step 2: Aligning with Word Corpus...")

    latent_dataset = torch.load(clip_latents_path)
    # 3b. Pre-initialize Aligner objects for each subject
    # We assume subject_corpora are in 'data/subject_corpora/' 
    # as generated by our previous script.
    subjects = sorted(list(set(item.get('subject') for item in latent_dataset)))
    aligners = {}

    for sub_id in subjects:
        corpus_path = os.path.join("data/subject_corpora", f"subject_{sub_id}_corpus.pt")
        if os.path.exists(corpus_path):
            print(f"  -> Loading Aligner for Subject {sub_id}...")
            aligners[sub_id] = Aligner(corpus_path, device=args.device)
        else:
            print(f"  [!] Warning: Corpus not found for Subject {sub_id} at {corpus_path}")

    # --- 3c. Calculate Noise PER SUBJECT (New logic) ---
    # Group all samples by subject ID to calculate their specific means
    print("Step 2b: Calculating subject-specific noise profiles...")
    subject_groups = defaultdict(list)
    for item in latent_dataset:
        subject_groups[item.get('subject')].append(item['eeg_clip_latent'])

    # Create a dictionary of mean noise vectors [1, 512] for each subject
    sub_noises = {
        sub_id: torch.stack(tensors).mean(dim=0).to(args.device) 
        for sub_id, tensors in subject_groups.items()
    }

    # --- 3d. Perform Per-Subject Alignment ---
    # --- Step 2: Subject-Specific Alignment with Dynamic Boosting ---
    
    
    aligned_results = []

    boost_mode = args.boost_mode

    for item in tqdm(latent_dataset, desc="Aligning + Boosting"):
        sub_id = item.get("subject")
        pred_obj = item.get("predicted_object_label", None) # The encoder's guess
        pred_conf = item.get("prediction_confidence", 0.0) # Confidence of the encoder's guess
        
        if sub_id in aligners:
            # Get subject-specific noise calculated in previous steps
            current_noise = sub_noises.get(sub_id)
            
            # Call Aligner with the predicted_object_label anchor
            bow_with_scores = aligners[sub_id].align(
                item['eeg_clip_latent'], 
                noise=current_noise,
                predicted_label=pred_obj,
                confidence=pred_conf,
                boost_mode=boost_mode,
                top_k=args.top_k
            )
            
            # Construct the detailed result object per your request
            aligned_results.append({
                "subject": sub_id,
                "gt_object_label": item.get("object_label", ""),   # Ground Truth
                "gt_caption": item.get("caption", ""),            # Ground Truth
                "predicted_object_label": pred_obj,               # Encoder Prediction
                "prediction_confidence": pred_conf,                 # Confidence Score
                "bow": bow_with_scores,                           # Boosted Results
                "prompt_words": [w['word'] for w in bow_with_scores]
            })

    # Final save to a tensor-compatible format
    aligned_path = os.path.join(args.output_dir, f"{dataset_name}_boosted_results_with_conf.pt")
    torch.save(aligned_results, aligned_path)
    print(f"✅ Step 2 Complete. Aligned results saved to {aligned_path}")

    if not args.skip_llm:
        print("Step 3: Generating Captions via LLM APIs...")
        boosted_results_path = "results/imagenet_eeg_test_boosted_results_with_conf.pt"
        latent_dataset = torch.load(boosted_results_path)
        

        # 3. Proceed with LLM Generation
        llm_manager = LLMManager(provider="openai", model_name="gpt-4o-mini-2024-07-18")
        final_output_path = os.path.join(args.output_dir, "all_subjects_feb20_linear_boost.pt")

        # Pass the sampled list directly to the experiment runner
        final_generations = llm_manager.run_decoding_experiment(
            input_path=boosted_results_path, # Update your method to accept this
            output_path=final_output_path
        )
        print(f"✅ Pipeline Complete. Samples saved to {final_output_path.replace('.pt', '.csv')}")
    # 5. Evaluation (To be implemented)
    if not args.skip_eval:
        print("Step 4: Running Evaluation Metrics (Coming Soon)...")

if __name__ == "__main__":
    main()