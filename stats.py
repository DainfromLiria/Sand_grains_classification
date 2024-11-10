from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pycocotools.coco import COCO

# Load COCO annotations
ann_file = "D:\\Study_AI\\BI-BAP\\Sand_grains_classification\\data\\annotations\\instances_default.json"
coco = COCO(ann_file)

# Extract category names and their respective counts
category_ids = coco.getCatIds()
categories = coco.loadCats(category_ids)
category_names = [cat['name'] for cat in categories]

# Get annotation counts for each category
annotations = coco.loadAnns(coco.getAnnIds())
category_counts = Counter([ann['category_id'] for ann in annotations])

# Map category IDs to category names
category_count_dict = {cat['name']: category_counts[cat['id']] for cat in categories}
print(sum(category_count_dict.values()))

# Create a pandas DataFrame for easier plotting
df = pd.DataFrame(list(category_count_dict.items()), columns=['Category', 'Count'])
df.sort_values('Count')


f, ax = plt.subplots(1, 1, figsize=(18, 7), sharex=True)
sns.barplot(x='Category', y='Count', data=df, palette="rocket", order=df.sort_values('Count', ascending=False).Category)
for i in ax.containers:
    ax.bar_label(i,)
ax.set_ylabel("Count", fontsize=15)
ax.set_xlabel("Label", fontsize=15)
ax.axes.set_title("Number of Annotations per Label", fontsize=20)
ax.tick_params(labelsize=11, rotation=75)
plt.show()
