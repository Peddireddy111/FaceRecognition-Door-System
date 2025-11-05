import os

dataset_path = "dataset"

for root, dirs, files in os.walk(dataset_path):
    print("📁 Current folder:", root)
    for f in files:
        print("   🖼️ File:", f)
