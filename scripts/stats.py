"""Script visualise statistics about original COCO dataset."""
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pycocotools.coco import COCO

# ENG to CZK dict
cat_dict = {
    'adhering_particles': 'přilnavé částice',
    'precipitation': 'sraženiny',
    'edge_abrasion': 'abraze hran',
    'pitting': 'tečkování',
    'conchoidal_fracture': 'lasturnatý lom',
}

# Load COCO annotations
# ann_file = "../data/train/annotations/instances_default.json"
ann_file = "../data/eval/annotations/instances_default.json"
coco = COCO(ann_file)
print("|==============================================================|")
print(f"Number of images: {len(coco.getImgIds())}")

# Extract category names and their counts
category_ids = coco.getCatIds()
categories = coco.loadCats(category_ids)
category_names = [cat['name'] for cat in categories]

# Get annotation counts for each category
annotations = coco.loadAnns(coco.getAnnIds())
category_counts = Counter([ann['category_id'] for ann in annotations])

category_count_dict = {cat_dict[cat['name']]: category_counts[cat['id']] for cat in categories}
print(f"Total count of features: {sum(category_count_dict.values())}")

df = pd.DataFrame(list(category_count_dict.items()), columns=['Category', 'Count'])
df.sort_values('Count')

# Make final barplot
f, ax = plt.subplots(1, 1, figsize=(18, 7), sharex=True)
sns.barplot(
    x='Category',
    y='Count',
    data=df,
    hue='Count',
    palette="viridis",
    order=df.sort_values('Count', ascending=False).Category,
    legend=False
)

ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=18)
ax.set_yticks(ax.get_yticks())
ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)

for i in ax.containers:
    ax.bar_label(i, fontsize=18)
ax.set_ylabel("Počet", fontsize=25)
ax.set_xlabel("Vlastnost", fontsize=25, labelpad=-30)

sns.despine()
plt.tight_layout()
plt.savefig("img/stats.png")
