import torch
import torch.nn as nn
from tqdm import tqdm
import os
import sys
from torch.utils.data import Dataset, DataLoader

import json
from safetensors.torch import load_file 

MODEL_REGISTRY = {
    "dream_diffusion": {"checkpoint": "models/dreamdiffusion_checkpoint.pth"},
    "channelnet": {
        "config": "models/config.json",
        "model": "models/model.safetensors"
    }
    # Add more encoders later
}
DATASET_REGISTRY = {
    "imagenet_eeg": "data/imagenet_eeg_text_dataset_all_subjects_5_95.pth",
    "imagenet_eeg_test": "data/eeg_55_95_text_dataset_test.pth",
    "imagenet_eeg_train": "data/eeg_55_95_text_dataset_train.pth"
}

sys.path.append(os.getcwd())
# # We assume sc_mbm is in your PYTHONPATH or the project root
# try:
#     from sc_mbm.mae_for_eeg import eeg_encoder 
# except ImportError:
#     print("Error: Ensure 'sc_mbm' is in your Python path.")


# class EEGDataset(Dataset):
#     def __init__(self, raw_data_list):
#         self.data_list = raw_data_list

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, idx):
#         item = self.data_list[idx]
#         # raw shape: [1, 128, 440] -> squeeze to [128, 440]
#         eeg_tensor = item['eeg_tensor'].squeeze(0)
        
#         # Metadata dictionary
#         meta = {
#             "caption": str(item.get("caption", "")),
#             "subject": str(item.get("subject", "unknown")),
#             "image_path": str(item.get("image_path", ""))
#         }
#         return eeg_tensor, meta

# class DreamDiffusionPipeline(nn.Module):
#     """Fuses MAE Encoder + Projection Layer into one unit."""
#     def __init__(self, checkpoint_path, device):
#         super().__init__()
#         self.device = device
        
#         # 1. Setup MAE Encoder (1024-D)
#         eeg_params = {
#             'time_len': 512, 'patch_size': 4, 'embed_dim': 1024,
#             'in_chans': 128, 'depth': 24, 'num_heads': 16, 'global_pool': False
#         }
#         self.mae_encoder = eeg_encoder(**eeg_params).to(device)
#         self.projector = nn.Linear(1024, 768).to(device)
        
#         sd = torch.load(checkpoint_path, map_location='cpu')['model_state_dict']
#         mae_sd = {k.replace('cond_stage_model.mae.', ''): v for k, v in sd.items() if 'cond_stage_model.mae.' in k}
#         self.mae_encoder.load_state_dict(mae_sd, strict=True)
        
#         proj_sd = {
#             'weight': sd['cond_stage_model.mapping.fc.weight'],
#             'bias': sd['cond_stage_model.mapping.fc.bias']
#         }
#         self.projector.load_state_dict(proj_sd, strict=True)
#         self.eval()

#     @torch.no_grad()
#     def forward(self, x):
#         return self.projector(self.mae_encoder(x))  # f(g(x))

# def process_dream_diffusion(dataset_path, output_path, device, batch_size=64):
#     cfg = MODEL_REGISTRY["dream_diffusion"]
#     pipe = DreamDiffusionPipeline(cfg["checkpoint"], device)
#     dataset = DATASET_REGISTRY[dataset_path]
#     raw_data = torch.load(dataset)
#     print(f"DEBUG: raw_data type: {type(raw_data)}")
#     print(f"DEBUG: raw_data length: {len(raw_data)}") # Is this 11,000 or 64?

#     dataset = EEGDataset(raw_data)
#     # loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
#     print(f"DEBUG: Number of batches in loader: {dataloader.__len__()}")  # Should be around 172 for 11,000 samples with batch size 64

#     final_dataset = []

#     print(f"Encoding {len(dataset)} EEG samples (Batch Size: {batch_size})...")

#     for batch_tensors, meta_list in tqdm(dataloader, desc="Encoding EEG to CLIP space"):
#         # batch tensors shape: [B, 128, 440] (after squeeze and before padding)

#         # 1. Move to device and pad in parallel 
#         eeg_input = batch_tensors.to(device)  # [B, 128, 440]
#         padding_needed = 512 - eeg_input.shape[2]
#         if padding_needed > 0:
#             eeg_input = nn.functional.pad(eeg_input, (0, padding_needed), 'constant', 0)  # [B, 128, 512]

#         # 2. Forward through pipeline
#         # eeg_input is now [B, 128, 512]
#         with torch.no_grad():
#             eeg_clip_latents = pipe(eeg_input)  # [B, 768]
#         print(f"DEBUG: eeg_clip_latents shape: {eeg_clip_latents.shape}")  # Should be [B, 768]
        
#         # 3. Collect results
#         eeg_clip_latents = eeg_clip_latents.cpu()
#         print(f"DEBUG: eeg_clip_latents moved to CPU, shape: {eeg_clip_latents.shape}")

#         # Zip back to dictionary 
#         for i in range(len(meta_list['caption'])):
#             final_dataset.append({
#                 "eeg_clip_latent": eeg_clip_latents[i],
#                 "caption": meta_list['caption'][i],
#                 "subject": meta_list['subject'][i],
#                 "image_path": meta_list['image_path'][i]
#             })
#         print(f"DEBUG: Processed batch of size {len(meta_list['caption'])}, total processed: {len(final_dataset)}")

#     print(f"Encoding complete. Total samples processed: {len(final_dataset)}. Saving to {output_path}...")
#     torch.save(final_dataset, output_path)    
#     return output_path


# Thought2Text ChannelNet EEG Encoder

from channelnet.model import ChannelNetModel
from channelnet.config import EEGModelConfig
from channelnet.constants import id2label
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd

class PreprocessedEEGDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        # Returns [1, 128, 440] tensor and metadata
        return item['eeg_tensor'], item['caption'], item['object_label'], item['subject']

@torch.no_grad()
def process_channelnet(dataset_path, output_path, device, batch_size):
    """
    Inference function for ChannelNet to extract CLIP-aligned latents.
    """
    config_path = MODEL_REGISTRY["channelnet"]["config"]
    model_path = MODEL_REGISTRY["channelnet"]["model"]
    # 1. Load Model & Config
    config = EEGModelConfig.from_json_file(config_path)
    model = ChannelNetModel.from_pretrained(model_path, config=config)
    model.to(device).eval()
    dataset = DATASET_REGISTRY[dataset_path]

    # 2. Prepare Data
    ds = PreprocessedEEGDataset(dataset)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    encoded_dataset = []
    print(f"Generating CLIP latents for {len(ds)} samples...")

    for eeg_tensors, captions, labels, subjects in tqdm(loader, desc="ChannelNet Encoding"):
        eeg_tensors = eeg_tensors.to(device)
        
        # Extract 512-D CLIP-aligned embedding (H_eeg) 
        # ChannelNet returns (embedding, classification_logits)
        embeddings, logits = model(eeg_tensors)
        probs = torch.nn.functional.softmax(logits, dim=1)
        confidences, preds = torch.max(probs, dim=1)

        
        for i in range(embeddings.size(0)):

            pred_idx_str = str(preds[i].item())
            pred_label = id2label.get(pred_idx_str, "unknown")

            encoded_dataset.append({
                "eeg_clip_latent": embeddings[i].cpu().unsqueeze(0),  # [512] -> [1, 512]
                "predicted_object_label": pred_label,
                "prediction_confidence": confidences[i].item(),
                "caption": captions[i],
                "object_label": labels[i],
                "subject": subjects[i].item()
            })

    torch.save(encoded_dataset, output_path)
    print(f"Saved {len(encoded_dataset)} encoded samples to {output_path}")