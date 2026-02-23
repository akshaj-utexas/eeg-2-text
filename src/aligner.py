import torch
# import torch.nn.functional as F
import torch.nn.functional as F
class Aligner:
    def __init__(self, corpus_path, device="cuda"):
        self.device = device
        data = torch.load(corpus_path)
        self.words = data["words"]
        self.word_to_idx = {w.lower(): i for i, w in enumerate(self.words)}
        self.word_embs = data['embeddings'].to(device).float() 
        
        # If the corpus shape is [Vocab, 1, 512], flatten it to [Vocab, 512]
        if self.word_embs.dim() == 3:
            self.word_embs = self.word_embs.squeeze(1) 

    @torch.no_grad()
    def align(self, eeg_latent, noise=None, predicted_label=None, confidence=None, boost_mode="linear", top_k=15):
        # 1. Ensure EEG latent is float32 and on correct device
        vec = eeg_latent.to(self.device).float()
        
        # 2. Global Noise Centering (if provided)
        if noise is not None:
            vec = vec - noise.to(self.device).float()
            
        # 3. Normalize for Cosine Similarity
        vec = F.normalize(vec, p=2, dim=-1)
        
        # 4. Similarity Search: [1, 512] @ [512, Vocab] -> [Vocab]
        # Using .squeeze() to ensure it's a 1D similarity vector
        sims = (vec @ self.word_embs.T).squeeze()


        # LOGARITHMIC BOOSTING LOGIC
        if predicted_label and confidence is not None:
            target_word = predicted_label.lower().strip()
            if target_word in self.word_to_idx:
                idx = self.word_to_idx[target_word]
                
                if boost_mode == "additive":
                    # Lambda controls the "gravity" of the prior
                    # Resulting scores may be negative; sims.topk() handles this
                    lambda_weight = 0.1 
                    sims[idx] += lambda_weight * torch.log(torch.tensor(confidence + 1e-6))
                
                elif boost_mode == "multiplicative":
                    # Beta controls the saturation curve
                    beta = 0.5
                    multiplier = 10.0
                    boost = 1 + beta * torch.log(1 + torch.tensor(confidence * multiplier))
                    sims[idx] *= boost

                elif boost_mode == "linear":
                    boost_factor = 2.5
                    idx = self.word_to_idx[target_word]
                    sims[idx] *= boost_factor
        
        # 5. Retrieve Top-K
        scores, indices = sims.topk(min(top_k, len(self.words)))
        
        return [
            {"word": self.words[i], "score": score.item()} 
            for score, i in zip(scores, indices)
        ]

def calculate_noise(dataset, device):
    """
    Computes Global Noise Centering across the dataset.
    dataset: List of dicts with 'eeg_clip_latent' shaped [1, 512]
    """
    # Simply concatenate the [1, 512] tensors into [N, 512]
    all_vecs = torch.cat([item['eeg_clip_latent'] for item in dataset], dim=0)
    # Return global mean [1, 512]
    return all_vecs.mean(dim=0, keepdim=True).to(device)