import os
from dotenv import load_dotenv
import openai
import torch
import anthropic
import google.generativeai as genai
import pandas as pd
from tqdm import tqdm

# Load variables from .env
load_dotenv()

class LLMManager:
    def __init__(self, provider="openai", model_name=None):
        self.provider = provider.lower()
        
        if self.provider == "openai":
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = model_name or "gpt-4o" # Using 4o for better reasoning
            
        elif self.provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model = model_name or "claude-3-5-sonnet-20240620"
            
        elif self.provider == "google":
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self.model_name = model_name or "gemini-1.5-pro"
            self.client = genai.GenerativeModel(self.model_name)

    def _build_prompt(self, sample):
        """
        Formats the words and their cosine similarities into a weighted prompt.
        word_scores is a list of dicts: [{'word': 'cat', 'score': 0.45}, ...]
        """
        # Sort by score descending (just in case)


        # 1. Extract inputs from our data structure
        pred_obj = sample.get("predicted_object_label", "unknown")
        # Softmax probability from our earlier encoder update
        pred_conf = sample.get("prediction_confidence", 0.0) 
        
        # 2. Extract and format the Bag of Words (BoW)
        # Handle different key naming conventions if necessary
        bow = sample.get("bow", []) or sample.get("bag_of_words", [])
        
        # Format the BoW string for the prompt
        weighted_list = [f"{item['word']} ({item['score']:.4f})" for item in bow]
        words_str = ", ".join(weighted_list)

        # 3. Construct your custom instruction-based prompt
        prompt = f"""You are given an object label and a noisy bag-of-words (BoW). Both object label and BoW will be accompanied with numbers, the numbers with object labels are the softmax probabilities of correctly guessing the object label, and the BoW are cosine similarities of the words to our embedding.

Your goal is to regenerate the most likely original image caption.

Instructions:
- Use the object label as a possible anchor.
- Use the similarity scores to infer which words are relevant.
- Ignore or remove garbage, irrelevant, contradictory, or low-signal words.
- Use only a small, coherent subset of the BoW plus the object label.
- Do NOT invent new objects not supported by the label or high-similarity words.

Output:
Return ONLY one natural-language caption (8–20 words). No explanations, no lists, no formatting.

Input:
Object label: {pred_obj} (prob: {pred_conf:.4f})

BoW tokens with scores:
{words_str}"""
        return prompt

    def generate(self, sample):
        prompt = self._build_prompt(sample)
        
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2 # Keep it grounded
                )
                return response.choices[0].message.content.strip().replace('"', '')

            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=100,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text.strip()

            elif self.provider == "google":
                response = self.client.generate_content(prompt)
                return response.text.strip()

        except Exception as e:
            return f"API ERROR ({self.provider}): {str(e)}"

    def run_decoding_experiment(self, input_path, output_path, num_samples=None):
        """
        Executes a batch decoding session and saves results.
        Call this directly from run_pipeline.py.
        """
        dataset = torch.load(input_path)
        test_subset = dataset[:num_samples]
        results = []

        print(f"🚀 Decoding {len(test_subset)} samples using {self.model}...")
        
        for item in tqdm(test_subset, desc="LLM Inference"):
            generated = self.generate(item)
            
            # Store everything for easy comparison
            results.append({
                "subject": item['subject'],
                "gt_object": item.get('gt_object_label', ''),
                "predicted_object": item.get('predicted_object_label', ''),
                "gt_caption": item.get('gt_caption', ''),
                "generated_caption": generated,
                "bow": [w['word'] for w in item['bow']]
            })

        # Save both formats
        torch.save(results, output_path)
        pd.DataFrame(results).to_csv(output_path.replace('.pt', '.csv'), index=False)
        print(f"✅ Saved results to {output_path} and CSV.")
        return results