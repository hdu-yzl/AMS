"""AntM2C step 2: encode the six textual fields with Multilingual BERT.

Following the AMS paper (Appendix A.1), AntM2C text embeddings are extracted
with Multilingual BERT (768-d). Each of the six text fields is encoded
independently per split and saved with a name that ``build_tfrecords.py``
expects, e.g. ``service_embeddings_train.npy``. The six 768-d vectors are
concatenated (in the order defined by ``ANT_TEXT_PARTS``) into the final 4608-d
text feature at TFRecord-build time.

Set ``BERT_PATH`` to a local directory or a HuggingFace model id.
"""

import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

FILE_PATHS = ["train.csv", "val.csv", "test.csv"]
SPLITS = ["train", "val", "test"]

# Multilingual BERT (paper Appendix A.1). Local folder by default, or a hub id.
BERT_PATH = os.environ.get("BERT_PATH", "pretrained_models/bert-base-multilingual-cased")

# Map each source column to the output file prefix used downstream.
COLUMN_TO_PREFIX = {
    "service_entity_seq": "service_embeddings",
    "query_entity_seq": "query_embeddings",
    "bill_entity_seq": "bill_embeddings",
    "item_entity_names": "item_entity_names_embeddings",
    "item_title": "item_title_embeddings",
    "log_time": "log_time_embeddings",
}


def encode_sequences(sequences, tokenizer, model, device, batch_size=256):
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc=f"Encoding on {device}"):
            batch = sequences[i: i + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(device)
            outputs = model(**inputs)
            embeddings.append(outputs.pooler_output.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    model = BertModel.from_pretrained(BERT_PATH).to(device)
    model.eval()

    for column, prefix in COLUMN_TO_PREFIX.items():
        for split, fp in zip(SPLITS, FILE_PATHS):
            data = pd.read_csv(fp, low_memory=False)
            sequences = data[column].fillna("").tolist()
            emb = encode_sequences(sequences, tokenizer, model, device)
            np.save(f"{prefix}_{split}.npy", emb)

    print("All text embeddings saved.")


if __name__ == "__main__":
    main()
