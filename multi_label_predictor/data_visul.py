import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from os.path import join as pjoin

# Example: Replace this with your actual dataset
# Assuming `labels` is a list or a pandas Series of class labels
# For example: labels = ["class1", "class1", "class2", "class3", "class3", "class3"]
annotated_dataset = pjoin(os.path.dirname(__file__), "combined_numpy_dataset", "annotated_numpy_data_encoded.npy")
np_dataset = np.load(annotated_dataset)
labels = np_dataset[:,:,2].unique()

# Count the number of occurrences of each class
class_counts = Counter(labels)

# If you are using a pandas DataFrame
# Example: df = pd.read_csv('your_dataset.csv')
# labels = df['label_column_name']
# class_counts = labels.value_counts()

# Convert to lists for visualization
classes = list(class_counts.keys())
counts = list(class_counts.values())

print(classes, counts)

# Plot the class distribution
plt.figure(figsize=(10, 6))
plt.bar(classes, counts, color='skyblue')
plt.xlabel('Classes', fontsize=14)
plt.ylabel('Number of Elements', fontsize=14)
plt.title('Class Distribution', fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

# Highlight imbalance if needed
for i, count in enumerate(counts):
    plt.text(i, count + 1, str(count), ha='center', fontsize=12)

plt.tight_layout()
plt.show()