import csv
import os


image_folder = "./training_data/images"
csv_file = "./training_data/labels.csv"

labels = {
    "img1.jpg": (170, 80), ## pranav
    "img2.jpg": (180, 65), ## domnic
    "img3.jpg": (175, 65), ## anurag
    "img4.jpg": (165, 60), ## neha
}

# Create CSV
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["image", "height", "weight"])  # header

    for img_name in os.listdir(image_folder):
        if img_name in labels:
            height, weight = labels[img_name]
            writer.writerow([img_name, height, weight])

print("Dataset CSV created successfully!")