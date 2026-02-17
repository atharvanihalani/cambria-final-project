"""Generate eval awareness graphs for scoping doc."""
import json
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# Load data
# ============================================================================
datasets = ['aime', 'hmmt_feb_2025', 'hmmt_nov_2025', 'olympmath']
dataset_labels = ['AIME\n2025', 'HMMT\nFeb 2025', 'HMMT\nNov 2025', 'OlympMATH']
dataset_labels_with_n = ['AIME\n2025\n(n=30)', 'HMMT\nFeb 2025\n(n=30)', 'HMMT\nNov 2025\n(n=30)', 'OlympMATH\n(n=200)']

original_eval_detection = []
rephrased_eval_detection = []
sharegpt_false_positive = []  # % of real convos incorrectly flagged as eval

for ds in datasets:
    with open(f'results/awareness/awareness_{ds}_all_PQs_claude-4.5-sonnet.json') as f:
        orig = json.load(f)['aggregated']
    with open(f'results/awareness/awareness_{ds}_rephrased_v3_all_PQs_claude-4.5-sonnet.json') as f:
        reph = json.load(f)['aggregated']

    original_eval_detection.append(orig['mean_accuracy_for_eval'] * 100)
    rephrased_eval_detection.append(reph['mean_accuracy_for_eval'] * 100)
    # false positive = 1 - accuracy_for_real (how often ShareGPT is wrongly called eval)
    sharegpt_false_positive.append((1 - orig['mean_accuracy_for_real']) * 100)

original_eval_detection = np.array(original_eval_detection)
rephrased_eval_detection = np.array(rephrased_eval_detection)
sharegpt_false_positive = np.array(sharegpt_false_positive)

# ============================================================================
# Shared style
# ============================================================================
BLUE = '#4878CF'
ORANGE = '#E8833A'
GREEN = '#6ACC65'
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.titleweight': 'bold',
    'figure.facecolor': 'white',
})

# ============================================================================
# Graph 1: Per-dataset eval detection – original vs rephrased
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 7))
fig.subplots_adjust(top=0.82)
x = np.arange(len(dataset_labels))
width = 0.35

bars1 = ax.bar(x - width/2, original_eval_detection, width, label='Original', color=BLUE)
bars2 = ax.bar(x + width/2, rephrased_eval_detection, width, label='Rephrased', color=ORANGE)

ax.set_ylabel('Eval Detection Rate (%)')
fig.suptitle('Eval Awareness by Dataset\n(Claude 4.5 Sonnet)', fontsize=15, fontweight='bold', y=0.95)
ax.set_xticks(x)
ax.set_xticklabels(dataset_labels_with_n)
ax.set_ylim(0, 100)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=2, frameon=False, fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('results/graph_awareness_by_dataset.png', dpi=150)
print('Saved: results/graph_awareness_by_dataset.png')
plt.close()

# ============================================================================
# Graph 2: Aggregate eval detection – original vs rephrased
# ============================================================================
fig, ax = plt.subplots(figsize=(6, 7))

agg_orig = original_eval_detection.mean()
agg_reph = rephrased_eval_detection.mean()
delta = agg_reph - agg_orig

bars = ax.bar(['Original', 'Rephrased'], [agg_orig, agg_reph], color=[BLUE, ORANGE], width=0.5)

ax.set_ylabel('Eval Detection Rate (%)')
ax.set_title('Aggregate Eval Awareness\n(Mean Across 4 Datasets)', pad=20)
ax.set_ylim(0, 100)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')


plt.tight_layout()
plt.savefig('results/graph_awareness_aggregate.png', dpi=150)
print('Saved: results/graph_awareness_aggregate.png')
plt.close()

# ============================================================================
# Graph 3: ShareGPT false positive rate vs rephrased eval detection
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(dataset_labels))
width = 0.35

bars1 = ax.bar(x - width/2, sharegpt_false_positive, width,
               label='ShareGPT False Positive Rate\n(real convos flagged as eval)', color=GREEN)
bars2 = ax.bar(x + width/2, rephrased_eval_detection, width,
               label='Rephrased Eval Detection\n(rephrased evals flagged as eval)', color=ORANGE)

ax.set_ylabel('Rate (%)')
ax.set_title('Rephrased Evals Look Like Real Conversations\n(Claude 4.5 Sonnet)')
ax.set_xticks(x)
ax.set_xticklabels(dataset_labels_with_n)
ax.set_ylim(0, 100)
ax.legend(loc='upper right', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('results/graph_awareness_baseline_vs_rephrased.png', dpi=150)
print('Saved: results/graph_awareness_baseline_vs_rephrased.png')
plt.close()

# ============================================================================
# Graph 4: Aggregate baseline vs rephrased
# ============================================================================
fig, ax = plt.subplots(figsize=(6, 6))

agg_fp = sharegpt_false_positive.mean()
agg_reph2 = rephrased_eval_detection.mean()

bars = ax.bar(
    ['ShareGPT\nFalse Positive', 'Rephrased\nEval Detection'],
    [agg_fp, agg_reph2],
    color=[GREEN, ORANGE],
    width=0.5,
)

ax.set_ylabel('Rate (%)')
ax.set_title('Aggregate: Rephrased Evals\nvs Real Conversation Baseline')
ax.set_ylim(0, 100)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('results/graph_awareness_baseline_vs_rephrased_aggregate.png', dpi=150)
print('Saved: results/graph_awareness_baseline_vs_rephrased_aggregate.png')
plt.close()

print('\nDone! All 4 graphs saved.')
