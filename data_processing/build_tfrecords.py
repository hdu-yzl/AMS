"""Build sequence-aware TFRecords for Amazon or AntM2C in a single pass.

This is the final, shared data-processing stage for both benchmarks. Given the
per-split CSV files and the pre-extracted ``.npy`` text/image feature files, it:

1. builds each user's chronological click history (label == 1 only);
2. attaches the last ``SEQ_LEN`` clicked item ids to every sample;
3. writes one TFRecord per (split, domain);
4. optionally shuffles the training shards.

Usage:
    python build_tfrecords.py --dataset amazon  --csv-dir split_data --feature-dir data --output-dir data
    python build_tfrecords.py --dataset antm2c  --csv-dir data       --feature-dir data --output-dir data
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


SPLITS = ("train", "val", "test")
SEQ_LEN = 5
SEED = 2025


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    csv_columns: Sequence[str]
    time_col: str
    domain_col: str
    domains: int
    text_dim: int
    image_dim: int
    min_item_id: int | None
    default_csv_dir: str
    default_feature_dir: str
    default_output_dir: str
    amazon_text_file: str = "text_features.npy"


DATASETS: Mapping[str, DatasetConfig] = {
    "amazon": DatasetConfig(
        name="amazon",
        csv_columns=(
            "user_id", "item_id", "domain", "label",
            "timestamp", "original_item_id", "original_user_id",
        ),
        time_col="timestamp",
        domain_col="domain",
        domains=3,
        text_dim=384,
        image_dim=4096,
        min_item_id=None,  # inferred from the data
        default_csv_dir="split_data",
        default_feature_dir="data",
        default_output_dir="data",
    ),
    "antm2c": DatasetConfig(
        name="antm2c",
        csv_columns=(
            "user_id", "item_id", "original_item_id", "log_time", "label",
            "bill_entity_seq", "service_entity_seq", "query_entity_seq",
            "item_entity_names", "item_title", "scene",
        ),
        time_col="log_time",
        domain_col="scene",
        domains=5,
        text_dim=768 * 6,
        image_dim=512,
        min_item_id=67625,
        default_csv_dir="data",
        default_feature_dir="data",
        default_output_dir="data",
    ),
}


# AntM2C concatenates six text embeddings per example, in this exact order.
ANT_TEXT_PARTS = (
    "service_embeddings_{split}.npy",
    "query_embeddings_{split}.npy",
    "bill_embeddings_{split}.npy",
    "item_entity_names_embeddings_{split}.npy",
    "item_title_embeddings_{split}.npy",
    "log_time_embeddings_{split}.npy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build seq-aware TFRecords for Amazon or AntM2C.")
    parser.add_argument("--dataset", choices=DATASETS.keys(), required=True)
    parser.add_argument("--csv-dir", default=None, help="directory with train/val/test.csv")
    parser.add_argument("--feature-dir", default=None, help="directory with .npy feature files")
    parser.add_argument("--output-dir", default=None, help="directory to write TFRecords")
    parser.add_argument("--shuffle-train", action="store_true", default=True)
    parser.add_argument("--no-shuffle-train", dest="shuffle_train", action="store_false")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--shuffle-buffer", type=int, default=50_000_000)
    parser.add_argument("--shard-size", type=int, default=500_000)
    return parser.parse_args()


def read_split(csv_dir: str, split: str, columns: Sequence[str]) -> pd.DataFrame:
    """Read a split CSV, tolerating both header and headerless layouts."""
    path = os.path.join(csv_dir, f"{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path, low_memory=False)
    if list(df.columns)[: len(columns)] != list(columns):
        df = pd.read_csv(path, header=None, names=columns, low_memory=False)
        if len(df) > 0 and list(df.iloc[0].astype(str))[: len(columns)] == list(columns):
            df = df.iloc[1:].reset_index(drop=True)
    return df


def normalize_types(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    """Coerce id/label/domain columns to numeric and drop invalid rows."""
    df = df.copy()
    for col in ("user_id", "item_id", "label", cfg.domain_col):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["user_id", "item_id", "label", cfg.domain_col, cfg.time_col])
    df["user_id"] = df["user_id"].astype(np.int64)
    df["item_id"] = df["item_id"].astype(np.int64)
    df["label"] = df["label"].astype(np.float32)
    df[cfg.domain_col] = df[cfg.domain_col].astype(np.int64)
    return df.reset_index(drop=True)


def load_splits(csv_dir: str, cfg: DatasetConfig) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        df = read_split(csv_dir, split, cfg.csv_columns).reset_index(drop=True)
        # Tag the original positional row index. AntM2C text embeddings are
        # stored per-row (one .npy row per CSV row), so we must index them by
        # the *original* position even after invalid rows are dropped.
        df["_orig_row"] = np.arange(len(df))
        out[split] = normalize_types(df, cfg)
    return out


def build_user_history(split_data: Mapping[str, pd.DataFrame], cfg: DatasetConfig) -> Dict[str, Tuple[List[str], List[str]]]:
    """Build each user's chronologically-ordered (items, labels) history."""
    data = pd.concat(split_data.values(), ignore_index=True)

    if cfg.name == "amazon":
        data["_seq_time"] = pd.to_datetime(data[cfg.time_col], unit="s", errors="coerce")
    else:
        data["_seq_time"] = pd.to_datetime(data[cfg.time_col], errors="coerce")

    data = data.dropna(subset=["user_id", "item_id", "label", "_seq_time"])
    data = data.sort_values(["user_id", "_seq_time"])

    history: Dict[str, Tuple[List[str], List[str]]] = {}
    for user_id, group in data.groupby("user_id", sort=False):
        items = group["item_id"].astype(np.int64).astype(str).tolist()
        labels = group["label"].astype(np.int64).astype(str).tolist()
        history[str(int(user_id))] = (items, labels)
    return history


def previous_clicked_items(user_id: int, item_id: int, history: Mapping[str, Tuple[List[str], List[str]]]) -> List[int]:
    """Return the last ``SEQ_LEN`` clicked item ids before the target item."""
    items, labels = history.get(str(int(user_id)), ([], []))
    target = str(int(item_id))
    try:
        pos = items.index(target)
    except ValueError:
        return [0] * SEQ_LEN

    clicked = [int(items[i]) for i in range(pos) if labels[i] == "1"]
    seq = clicked[-SEQ_LEN:]
    return [0] * (SEQ_LEN - len(seq)) + seq


def feature_index(features: np.ndarray, item_id: int, min_item_id: int) -> int:
    """Map an item id to a row in the feature matrix, honouring the pad row."""
    raw_idx = int(item_id) - min_item_id
    shifted_idx = raw_idx + 1
    has_zero_row = len(features) > 0 and np.allclose(features[0], 0)
    if has_zero_row and shifted_idx < len(features):
        return shifted_idx
    return raw_idx


def tf_feature(text_features, image_features, id_features, label, domain, user_seq) -> bytes:
    feature = {
        "text_features": tf.train.Feature(
            float_list=tf.train.FloatList(value=np.asarray(text_features, dtype=np.float32).ravel())
        ),
        "image_features": tf.train.Feature(
            float_list=tf.train.FloatList(value=np.asarray(image_features, dtype=np.float32).ravel())
        ),
        "id_feature": tf.train.Feature(int64_list=tf.train.Int64List(value=list(id_features))),
        "label": tf.train.Feature(float_list=tf.train.FloatList(value=[float(label)])),
        "domain": tf.train.Feature(float_list=tf.train.FloatList(value=[float(domain)])),
        "user_seq": tf.train.Feature(int64_list=tf.train.Int64List(value=list(user_seq))),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


class FeatureProvider:
    """Loads text/image features and serves them per sample for each dataset."""

    def __init__(self, cfg: DatasetConfig, split_data: Mapping[str, pd.DataFrame], feature_dir: str, output_dir: str):
        self.cfg = cfg
        self.feature_dir = feature_dir
        self.output_dir = output_dir
        self.image_features = np.load(os.path.join(feature_dir, "image_feature.npy"))

        all_items = pd.concat([df["item_id"] for df in split_data.values()], ignore_index=True)
        self.min_item_id = cfg.min_item_id or int(all_items.min())

        self.amazon_text_features: np.ndarray | None = None
        self.ant_text_parts: Dict[str, List[np.ndarray]] = {}
        self.ant_item_title = None

        if cfg.name == "amazon":
            self.amazon_text_features = np.load(os.path.join(feature_dir, cfg.amazon_text_file))
        else:
            self.ant_text_parts = {
                split: [np.load(os.path.join(feature_dir, p.format(split=split))) for p in ANT_TEXT_PARTS]
                for split in SPLITS
            }
            # Persist the item-title lookup matrix used by the sequence loader.
            self.ant_item_title = self._build_ant_item_title_matrix(split_data)
            np.save(os.path.join(output_dir, "text_feature.npy"), self.ant_item_title)

    def text(self, split: str, row_idx: int, item_id: int) -> np.ndarray:
        if self.cfg.name == "amazon":
            assert self.amazon_text_features is not None
            return self.amazon_text_features[feature_index(self.amazon_text_features, item_id, self.min_item_id)]
        return np.concatenate([part[row_idx] for part in self.ant_text_parts[split]])

    def image(self, item_id: int) -> np.ndarray:
        return self.image_features[feature_index(self.image_features, item_id, self.min_item_id)]

    def _build_ant_item_title_matrix(self, split_data: Mapping[str, pd.DataFrame]) -> np.ndarray:
        all_items = pd.concat([df["item_id"] for df in split_data.values()], ignore_index=True)
        item_count = int(all_items.max()) - self.min_item_id + 1
        matrix = np.zeros((item_count + 1, 768), dtype=np.float32)  # row 0 = pad

        for split, df in split_data.items():
            title_embeddings = self.ant_text_parts[split][4]  # item_title is index 4
            # Index embeddings by the original CSV row position (see load_splits).
            for orig_row, item_id in zip(df["_orig_row"].astype(int), df["item_id"].astype(np.int64)):
                out_idx = int(item_id) - self.min_item_id + 1
                if 0 <= out_idx < len(matrix) and not matrix[out_idx].any():
                    matrix[out_idx] = title_embeddings[int(orig_row)]
        return matrix


def write_domain_tfrecords(split, df, cfg, features, history, output_dir) -> None:
    writers = [
        tf.io.TFRecordWriter(os.path.join(output_dir, f"{split}{domain}.tfrecord"))
        for domain in range(cfg.domains)
    ]
    counts = [0] * cfg.domains

    try:
        for row_idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{cfg.name}:{split}"):
            domain = int(row[cfg.domain_col])
            if domain < 0 or domain >= cfg.domains:
                raise ValueError(f"{split} row {row_idx} has invalid domain {domain}")

            user_id = int(row["user_id"])
            item_id = int(row["item_id"])
            orig_row = int(row["_orig_row"])
            record = tf_feature(
                text_features=features.text(split, orig_row, item_id),
                image_features=features.image(item_id),
                id_features=(user_id, item_id),
                label=float(row["label"]),
                domain=domain,
                user_seq=previous_clicked_items(user_id, item_id, history),
            )
            writers[domain].write(record)
            counts[domain] += 1
    finally:
        for writer in writers:
            writer.close()

    print(f"{split} domain counts: {counts}")


def shuffle_train_records(output_dir, data_type, seed, buffer_size, shard_size) -> None:
    files = sorted(glob.glob(os.path.join(output_dir, f"{data_type}[0-9].tfrecord")))
    if not files:
        raise FileNotFoundError(f"No files matched {data_type}[0-9].tfrecord in {output_dir}")

    dataset = tf.data.TFRecordDataset(files).shuffle(
        buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=False
    )

    writer = None
    shard_idx = 0
    sample_count = 0
    try:
        for raw_record in dataset.as_numpy_iterator():
            if sample_count % shard_size == 0:
                if writer is not None:
                    writer.close()
                out_file = os.path.join(output_dir, f"{data_type}_shuffle_{shard_idx}.tfrecord")
                writer = tf.io.TFRecordWriter(out_file)
                print(f"Writing {out_file}")
                shard_idx += 1
            writer.write(raw_record)
            sample_count += 1
    finally:
        if writer is not None:
            writer.close()

    print(f"Shuffled {sample_count} {data_type} samples into {shard_idx} shard(s).")


def main() -> None:
    args = parse_args()
    cfg = DATASETS[args.dataset]
    csv_dir = args.csv_dir or cfg.default_csv_dir
    feature_dir = args.feature_dir or cfg.default_feature_dir
    output_dir = args.output_dir or cfg.default_output_dir
    os.makedirs(output_dir, exist_ok=True)

    split_data = load_splits(csv_dir, cfg)
    history = build_user_history(split_data, cfg)
    features = FeatureProvider(cfg, split_data, feature_dir, output_dir)

    for split in SPLITS:
        write_domain_tfrecords(split, split_data[split], cfg, features, history, output_dir)

    if args.shuffle_train:
        shuffle_train_records(
            output_dir=output_dir,
            data_type="train",
            seed=args.seed,
            buffer_size=args.shuffle_buffer,
            shard_size=args.shard_size,
        )


if __name__ == "__main__":
    main()
