"""Amazon step 4: extract item text and image embeddings.

Text embeddings are produced with a Sentence-Transformers MiniLM encoder over a
concatenation of title / brand / category / description. Image embeddings are
read from the pre-computed Amazon visual feature binaries (4096-d each).

The resulting matrices are indexed by ``item_id - min_item_id``:
    data/text_features.npy   (num_items, 384)
    data/image_feature.npy   (num_items, 4096)

Set ``TEXT_ENCODER_PATH`` to a local path or a HuggingFace model id.
"""

import os
import array

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

FILE_PATHS = [
    "split_data/train.csv",
    "split_data/val.csv",
    "split_data/test.csv",
]
TEXT_FILE = "split_data/item_meta.csv"
IMAGE_FEATURE_FILES = [
    "data/image_features_Sports_and_Outdoors.b",
    "data/image_features_Movies_and_TV.b",
    "data/image_features_Books.b",
]

# Sentence-Transformers MiniLM (paper Appendix A.1). Local folder or a hub id.
TEXT_ENCODER_PATH = os.environ.get("TEXT_ENCODER_PATH", "pretrained_models/all-MiniLM-L6-v2")
IMAGE_DIM = 4096
TEXT_DIM = 384


def load_item_id_mapping():
    frames = [pd.read_csv(fp, low_memory=False) for fp in FILE_PATHS]
    all_data = pd.concat(frames)
    return all_data.set_index("original_item_id")["item_id"].to_dict()


def read_image_features(path):
    """Yield (asin, 4096-d feature) pairs from a binary visual feature file."""
    with open(path, "rb") as f:
        while True:
            asin = f.read(10).decode("UTF-8")
            if asin == "":
                break
            a = array.array("f")
            a.fromfile(f, IMAGE_DIM)
            yield asin, a.tolist()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class TextDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length=512):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.sentences[idx], padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in inputs.items()}


def build_sentences(df):
    """Concatenate the textual metadata fields into a single sentence per item."""
    sentences = []
    for _, row in df.iterrows():
        sen = row["title"] + " " + row["brand"] + " "
        cates = eval(row["categories"])
        if isinstance(cates, list):
            for c in cates[0]:
                sen = sen + c + " "
        sen += row["description"]
        sentences.append(sen.replace("\n", " "))
    return sentences


def main():
    item_id_mapping = load_item_id_mapping()

    df = pd.read_csv(TEXT_FILE, low_memory=False)
    for col in ("title", "brand", "categories", "description"):
        df[col] = df[col].fillna(" ")

    tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER_PATH)
    model = AutoModel.from_pretrained(TEXT_ENCODER_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    sentences = build_sentences(df)
    dataloader = DataLoader(TextDataset(sentences, tokenizer), batch_size=16, shuffle=False)

    all_embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            model_output = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_embeddings = mean_pooling(model_output, attention_mask)
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            all_embeddings.append(batch_embeddings.cpu().numpy())
    sentence_embedding = np.concatenate(all_embeddings, axis=0)

    min_item_id = df["itemID"].min()
    max_item_id = df["itemID"].max()

    # Text feature matrix indexed by (itemID - min_item_id).
    text_matrix = np.zeros((max_item_id - min_item_id + 1, TEXT_DIM), dtype=np.float32)
    for i, embedding in enumerate(sentence_embedding):
        text_matrix[df["itemID"].iloc[i] - min_item_id] = embedding
    np.save("data/text_features.npy", text_matrix)
    print("Text features saved to data/text_features.npy")

    # Image feature matrix; missing items fall back to the dataset mean vector.
    image_matrix = np.zeros((max_item_id - min_item_id + 1, IMAGE_DIM), dtype=np.float32)
    feats, avg = {}, []
    for img_file in IMAGE_FEATURE_FILES:
        for asin, feat in read_image_features(img_file):
            if asin in item_id_mapping:
                feats[item_id_mapping[asin] - min_item_id] = feat
                avg.append(feat)
    avg = np.array(avg).mean(0).tolist()
    for i in range(max_item_id - min_item_id + 1):
        image_matrix[i] = feats[i] if i in feats else avg
    np.save("data/image_feature.npy", np.array(image_matrix, dtype=np.float32))
    print("Image features saved to data/image_feature.npy")


if __name__ == "__main__":
    main()
