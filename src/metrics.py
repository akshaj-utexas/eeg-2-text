import torch
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm

# Ensure NLTK resources are ready
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

def calculate_metrics(results_path):
    """
    Calculates detailed NLP metrics and generates presentation-ready wide plots.
    """
    # 1. Load the data
    print(f"Loading results from {results_path}...")
    data = torch.load(results_path)
    
    # 2. Setup Scorer (Added ROUGE-1 and ROUGE-2)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smooth = SmoothingFunction().method1
    
    all_metrics = []

    print("Computing metrics for all samples...")
    for item in tqdm(data):
        gt = item['gt_caption'].lower().strip()
        gen = item['generated_caption'].lower().strip()
        
        gt_tokens = nltk.word_tokenize(gt)
        gen_tokens = nltk.word_tokenize(gen)

        # BLEU Calculations
        bleu1 = sentence_bleu([gt_tokens], gen_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
        bleu4 = sentence_bleu([gt_tokens], gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
        
        # ROUGE Calculations
        rs = scorer.score(gt, gen)
        r1 = rs['rouge1'].fmeasure
        r2 = rs['rouge2'].fmeasure
        rl = rs['rougeL'].fmeasure
        
        # METEOR
        met = meteor_score([gt_tokens], gen_tokens)

        all_metrics.append({
            "subject": int(item['subject']), # Cast to int for sorting
            "bleu1": bleu1,
            "bleu4": bleu4,
            "rouge1": r1,
            "rouge2": r2,
            "rougeL": rl,
            "meteor": met
        })

    # 3. Aggregate and Process
    df = pd.DataFrame(all_metrics).sort_values('subject')
    metrics_cols = ['bleu1', 'bleu4', 'rouge1', 'rouge2', 'rougeL', 'meteor']
    subject_summary = df.groupby('subject')[metrics_cols].mean() * 100
    overall_summary = df[metrics_cols].mean() * 100

    # 4. Print Numbers for the Meeting
    print("\n" + "="*90)
    print("      📊 STRATIFIED DECODING PERFORMANCE: SUBJECT-WISE SUMMARY (%)")
    print("="*90)
    fmt = {c: '{:0.2f}'.format for c in metrics_cols}
    print(subject_summary.to_string(formatters=fmt))
    print("-" * 90)
    print(f"OVERALL AVG | B1: {overall_summary['bleu1']:.2f} | B4: {overall_summary['bleu4']:.2f} | R1: {overall_summary['rouge1']:.2f} | R2: {overall_summary['rouge2']:.2f} | RL: {overall_summary['rougeL']:.2f} | MET: {overall_summary['meteor']:.2f}")
    print("="*90)

    # 5. GENERATE PRESENTATION PLOTS
    print("Creating visualization suite...")
    sns.set_theme(style="whitegrid")
    
    # We will create 3 plots: BLEU, ROUGE, and METEOR (wide aspect)
    plot_data = subject_summary.reset_index()
    
    # Setup Figure with 3 subplots vertically to allow for WIDE layout
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))
    plt.subplots_adjust(hspace=0.4)

    # --- Plot A: BLEU Metrics (Precision) ---
    bleu_df = plot_data.melt(id_vars='subject', value_vars=['bleu1', 'bleu4'], var_name='Metric', value_name='Score')
    sns.barplot(ax=axes[0], data=bleu_df, x='subject', y='Score', hue='Metric', palette='Blues_d')
    axes[0].set_title("BLEU Performance: Unigrams (B1) vs Quadgrams (B4)", fontsize=16, fontweight='bold')
    axes[0].set_ylabel("Score (%)", fontsize=12)

    # --- Plot B: ROUGE Metrics (Recall/Sequence) ---
    rouge_df = plot_data.melt(id_vars='subject', value_vars=['rouge1', 'rouge2', 'rougeL'], var_name='Metric', value_name='Score')
    sns.barplot(ax=axes[1], data=rouge_df, x='subject', y='Score', hue='Metric', palette='Greens_d')
    axes[1].set_title("ROUGE Performance: Content Overlap (R1, R2, RL)", fontsize=16, fontweight='bold')
    axes[1].set_ylabel("Score (%)", fontsize=12)

    # --- Plot C: METEOR (Semantic Alignment) ---
    sns.barplot(ax=axes[2], data=plot_data, x='subject', y='meteor', color='salmon', alpha=0.8)
    axes[2].set_title("METEOR Performance: Semantic & Synonym Matching", fontsize=16, fontweight='bold')
    axes[2].set_ylabel("Score (%)", fontsize=12)

    # Final touches to all plots
    for ax in axes:
        ax.set_xlabel("Subject ID", fontsize=12)
        ax.set_ylim(0, max(overall_summary) + 15) # Dynamic ceiling
        # Add labels on top of bars
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f'{p.get_height():.1f}%', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='center', fontsize=10, color='black', xytext=(0, 8), 
                            textcoords='offset points')

    # Save
    plot_path = results_path.replace('.pt', '_full_presentation_suite.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Presentation suite saved to {plot_path}")
    
    # Save CSV
    df.to_csv(results_path.replace('.pt', '_full_meeting_metrics.csv'), index=False)

if __name__ == "__main__":
    calculate_metrics("results/all_subjects_feb20_linear_boost.pt")

# import torch
# import pandas as pd
# import nltk
# import matplotlib.pyplot as plt
# import seaborn as sns
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from nltk.translate.meteor_score import meteor_score
# from rouge_score import rouge_scorer
# from tqdm import tqdm

# # Ensure NLTK resources are ready
# nltk.download('punkt', quiet=True)
# nltk.download('wordnet', quiet=True)

# def calculate_metrics(results_path):
#     """
#     Calculates BLEU-1, BLEU-3, and other metrics from the decoded results and generates a plot.
#     """
#     # 1. Load the data
#     print(f"Loading results from {results_path}...")
#     data = torch.load(results_path)
    
#     # 2. Setup Scorer
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
#     smooth = SmoothingFunction().method1
    
#     all_metrics = []

#     print("Computing metrics for all samples...")
#     for item in tqdm(data):
#         gt = item['gt_caption'].lower().strip()
#         gen = item['generated_caption'].lower().strip()
        
#         gt_tokens = nltk.word_tokenize(gt)
#         gen_tokens = nltk.word_tokenize(gen)

#         # NLP Metrics calculation
#         bleu1 = sentence_bleu([gt_tokens], gen_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
#         bleu3 = sentence_bleu([gt_tokens], gen_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
#         bleu4 = sentence_bleu([gt_tokens], gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
#         rougeL = scorer.score(gt, gen)['rougeL'].fmeasure
#         met = meteor_score([gt_tokens], gen_tokens)

#         all_metrics.append({
#             "subject": item['subject'],
#             "bleu1": bleu1,
#             "bleu3": bleu3,
#             "bleu4": bleu4,
#             "rougeL": rougeL,
#             "meteor": met
#         })

#     # 3. Aggregate Subject-Wise Stats
#     df = pd.DataFrame(all_metrics)
#     metrics_cols = ['bleu1', 'bleu3', 'bleu4', 'rougeL', 'meteor']
#     subject_summary = df.groupby('subject')[metrics_cols].mean() * 100
#     overall_summary = df[metrics_cols].mean() * 100

#     # 4. Final Formatted Report
#     print("\n" + "="*80)
#     print("      📊 SUBJECT-WISE PERFORMANCE REPORT (%)")
#     print("="*80)
#     fmt = {c: '{:0.2f}'.format for c in metrics_cols}
#     print(subject_summary.to_string(formatters=fmt))
#     print("-" * 80)
#     print(f"OVERALL AVG | B1: {overall_summary['bleu1']:.2f} | B3: {overall_summary['bleu3']:.2f} | B4: {overall_summary['bleu4']:.2f} | RL: {overall_summary['rougeL']:.2f} | MET: {overall_summary['meteor']:.2f}")
#     print("="*80)

#     # 5. GENERATE PLOT
#     print("Generating subject-wise performance plot...")
#     sns.set_theme(style="whitegrid")
    
#     # Reshape data for plotting
#     plot_df = subject_summary.reset_index().melt(
#         id_vars='subject', 
#         var_name='Metric', 
#         value_name='Score (%)'
#     )
    
#     # Create the grouped bar chart
#     # Use a clear palette and ensure bars are sorted by subject ID
#     ax = sns.barplot(
#         data=plot_df, 
#         x='subject', 
#         y='Score (%)', 
#         hue='Metric', 
#         palette='viridis'
#     )
    
#     # Formatting
#     plt.title("Subject-wise Decoding Performance Across Metrics", fontsize=14, pad=15)
#     plt.xlabel("Subject ID", fontsize=12)
#     plt.ylabel("Average Score (%)", fontsize=12)
#     plt.legend(title="NLP Metric", bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
    
#     # Save the figure
#     plot_path = results_path.replace('.pt', '_performance_plot.png')
#     plt.savefig(plot_path, dpi=300)
#     print(f"✅ Plot saved to {plot_path}")
    
#     # Save the expanded metrics CSV
#     csv_path = results_path.replace('.pt', '_expanded_metrics.csv')
#     df.to_csv(csv_path, index=False)
#     print(f"✅ Detailed metrics saved to {csv_path}")

# if __name__ == "__main__":
#     calculate_metrics("results/feb_20_final_generations_stratified.pt")