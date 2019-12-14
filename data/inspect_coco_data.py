import json
import matplotlib.pyplot as plt
from imantics import Dataset

if __name__ == '__main__':
    with open('composition_dataset_2/train/annotations/annotations.json') as f:
        data = json.load(f)

    dataset = Dataset.from_coco(data)
    for image in dataset.iter_images():
        draw = image.draw(bbox=True)
        plt.imshow(draw)
        plt.show()
     