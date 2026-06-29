"""AntM2C step 3: extract item image embeddings with Chinese-CLIP.

Each item image (named by its ``original_item_id``) is encoded into a 512-d
vector. Missing images fall back to an all-zero vector. The matrix is indexed by
``item_id - min_item_id`` and saved as ``image_feature.npy``.

Set ``CLIP_PATH`` to a local directory or a HuggingFace model id.
"""

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

FILE_PATHS = ["train.csv", "val.csv", "test.csv"]
PICTURE_PATH = "AntM2C_image"
# Chinese-CLIP (paper Appendix A.1). Local folder by default, or a hub id.
CLIP_PATH = os.environ.get("CLIP_PATH", "pretrained_models/chinese-clip-vit-base-patch16")
IMAGE_DIM = 512


def get_item_id_to_original_id_mapping(data):
    item_to_original = {}
    for _, row in data.iterrows():
        item_id = row["item_id"]
        if item_id not in item_to_original:
            item_to_original[item_id] = row["original_item_id"]
    return item_to_original


def extract_image_features(item_ids, original_item_ids, model, preprocess, device):
    image_features = []
    for item_id in tqdm(item_ids, desc="Extracting image features"):
        original_id = original_item_ids[item_id]
        image_path = os.path.join(PICTURE_PATH, f"{original_id}.png")
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            print(f"Missing image for original_item_id: {original_id}")
            image_features.append(np.zeros(IMAGE_DIM))
            continue

        with torch.no_grad():
            inputs = preprocess(images=image, return_tensors="pt").to(device)
            outputs = model.get_image_features(**inputs)
            image_features.append(outputs.detach().cpu().numpy().flatten())
    return np.vstack(image_features).astype(np.float32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChineseCLIPModel.from_pretrained(CLIP_PATH).to(device)
    model.eval()
    preprocess = ChineseCLIPProcessor.from_pretrained(CLIP_PATH)

    all_data = pd.concat([pd.read_csv(fp, low_memory=False) for fp in FILE_PATHS])
    item_to_original = get_item_id_to_original_id_mapping(all_data)

    unique_item_ids = sorted(item_to_original.keys())
    print(f"Total unique item_ids: {len(unique_item_ids)}")

    image_features = extract_image_features(unique_item_ids, item_to_original, model, preprocess, device)

    min_item_id, max_item_id = min(unique_item_ids), max(unique_item_ids)
    print(f"Minimum item_id: {min_item_id}, Maximum item_id: {max_item_id}")

    feature_matrix = np.zeros((max_item_id - min_item_id + 1, image_features.shape[1]), dtype=np.float32)
    for idx, item_id in enumerate(unique_item_ids):
        feature_matrix[item_id - min_item_id] = image_features[idx]

    print(f"Feature matrix shape: {feature_matrix.shape}")
    np.save("image_feature.npy", feature_matrix)
    print("Image features saved to image_feature.npy")


if __name__ == "__main__":
    main()
