"""Unified TFRecord loaders for the Amazon and AntM2C benchmarks.

Every TFRecord example stores the following fields::

    text_features  : float[text_dim]    # item text embedding(s)
    image_features : float[image_dim]   # item image embedding
    id_feature     : int64[2]           # (user_id, item_id)
    label          : float[1]           # click label (0/1)
    domain         : float[1]           # domain / scene id
    user_seq       : int64[5]           # last 5 clicked item ids (left-padded with 0)

The loader yields PyTorch tensors and optionally gathers the per-step user
behaviour sequence features (``get_data_seq*``).
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import torch


@dataclass(frozen=True)
class LoaderConfig:
    """Per-dataset feature dimensions and on-disk layout."""

    text_features: int
    image_features: int
    min_item_id: int
    item_text_file: str
    item_image_file: str = "image_feature.npy"
    # AntM2C stores 6 concatenated 768-d text embeddings per example; the item
    # title sits at index 4. When True, ``get_data_seq*`` splits them apart.
    antm2c_text_layout: bool = False


CONFIGS = {
    "amazon": LoaderConfig(
        text_features=384,
        image_features=4096,
        min_item_id=143639,
        item_text_file="text_features.npy",
    ),
    "antm2c": LoaderConfig(
        text_features=768 * 6,
        image_features=512,
        min_item_id=67625,
        item_text_file="text_feature.npy",
        antm2c_text_layout=True,
    ),
}


class SeqTFRecordLoader:
    """Generic sequence-aware TFRecord loader shared by both datasets."""

    def __init__(self, tfrecord_path: str, config: LoaderConfig):
        self.samples = 1
        self.id_features = 2
        self.seq_len = 5
        self.text_features = config.text_features
        self.image_features = config.image_features
        self.min_item_id = config.min_item_id
        self.tfrecord_path = tfrecord_path
        self.config = config

        self.description = {
            "text_features": tf.io.FixedLenFeature([self.text_features], tf.float32),
            "image_features": tf.io.FixedLenFeature([self.image_features], tf.float32),
            "id_feature": tf.io.FixedLenFeature([self.id_features], tf.int64),
            "label": tf.io.FixedLenFeature([self.samples], tf.float32),
            "domain": tf.io.FixedLenFeature([self.samples], tf.float32),
            "user_seq": tf.io.FixedLenFeature([self.seq_len], tf.int64),
        }

        # Item-level lookup tables used to materialise the behaviour sequence.
        # Row 0 is reserved as the all-zero padding row.
        self.title_feat_torch = torch.from_numpy(
            self._load_zero_padded_matrix(config.item_text_file, fallback="item_text.npy")
        )
        self.image_feat_torch = torch.from_numpy(
            self._load_zero_padded_matrix(config.item_image_file)
        )

    def _load_zero_padded_matrix(self, filename: str, fallback: str | None = None) -> np.ndarray:
        """Load a feature matrix, ensuring its first row is an all-zero pad row."""
        path = os.path.join(self.tfrecord_path, filename)
        if not os.path.exists(path) and fallback is not None:
            path = os.path.join(self.tfrecord_path, fallback)
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        matrix = np.load(path)
        if len(matrix) > 0 and np.allclose(matrix[0], 0):
            return matrix

        zero = np.zeros((1, matrix.shape[1]), dtype=matrix.dtype)
        return np.concatenate([zero, matrix], axis=0)

    def _files(self, data_type: str, shuffled: bool = False, domain: int | None = None) -> list[str]:
        """Resolve the TFRecord files for a split / shuffle / single-domain query."""
        if domain is not None:
            pattern = f"{data_type}{domain}.tfrecord"
        elif shuffled:
            pattern = f"{data_type}_shuffle*.tfrecord"
        else:
            pattern = f"{data_type}[0-9].tfrecord"

        files = sorted(glob.glob(os.path.join(self.tfrecord_path, pattern)))
        if not files and not shuffled and domain is None:
            files = sorted(glob.glob(os.path.join(self.tfrecord_path, f"{data_type}.tfrecord")))
        if not files:
            raise FileNotFoundError(os.path.join(self.tfrecord_path, pattern))
        return files

    def _dataset(self, files: list[str], batch_size: int):
        def read_data(raw_rec):
            example = tf.io.parse_single_example(raw_rec, self.description)
            return (
                example["text_features"],
                example["image_features"],
                example["id_feature"],
                example["label"],
                example["domain"],
                example["user_seq"],
            )

        return (
            tf.data.TFRecordDataset(files)
            .map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

    def _iter_basic(self, files: list[str], batch_size: int):
        """Yield ``(id_feat, text, image, label, domain)`` tensors per batch."""
        for text, image, id_feat, label, domain, _user_seq in self._dataset(files, batch_size):
            yield (
                torch.from_numpy(id_feat.numpy()),
                torch.from_numpy(text.numpy()),
                torch.from_numpy(image.numpy()),
                torch.from_numpy(label.numpy()),
                torch.from_numpy(domain.numpy()),
            )

    def _iter_seq(self, files: list[str], batch_size: int):
        """Yield batches including the materialised behaviour-sequence features."""
        for text, image, id_feat, label, domain, user_seq in self._dataset(files, batch_size):
            text_t = torch.from_numpy(text.numpy())
            item_image = torch.from_numpy(image.numpy())
            id_feat_t = torch.from_numpy(id_feat.numpy())
            label_t = torch.from_numpy(label.numpy())
            domain_t = torch.from_numpy(domain.numpy())
            user_seq_t = torch.from_numpy(user_seq.numpy())

            if self.config.antm2c_text_layout:
                # Split the 6 concatenated text embeddings; index 4 is the item title.
                text_full = text_t.reshape(-1, 6, 768)
                user_text = torch.cat(
                    [text_full[:, 0:4, :], text_full[:, 5:6, :]], dim=1
                ).reshape(text_t.shape[0], -1)
                item_text = text_full[:, 4, :]
            else:
                item_text = text_t
                user_text = item_text  # placeholder for the single-text layout

            # Map item ids in the behaviour sequence to lookup-table rows
            # (0 stays 0 because it is the padding row).
            idx = torch.where(user_seq_t == 0, 0, user_seq_t - self.min_item_id + 1)
            item_text_seq = self.title_feat_torch[idx]
            item_image_seq = self.image_feat_torch[idx]

            yield (
                id_feat_t,
                user_text,
                item_text,
                item_image,
                label_t,
                domain_t,
                user_seq_t,
                item_text_seq,
                item_image_seq,
            )

    # --- Public API -------------------------------------------------------

    def get_data(self, data_type: str, batch_size: int = 512):
        """Iterate over a split (``train`` / ``val`` / ``test``)."""
        yield from self._iter_basic(self._files(data_type), batch_size)

    def get_data_shuffle(self, data_type: str, batch_size: int = 512):
        """Iterate over the pre-shuffled shards of a split."""
        yield from self._iter_basic(self._files(data_type, shuffled=True), batch_size)

    def get_data_split_domain(self, data_type: str, batch_size: int = 512, domain: int = 0):
        """Iterate over a single domain of a split."""
        yield from self._iter_basic(self._files(data_type, domain=domain), batch_size)

    def get_data_seq(self, data_type: str, batch_size: int = 512):
        """Iterate over a split with behaviour-sequence features attached."""
        yield from self._iter_seq(self._files(data_type), batch_size)

    def get_data_seq_shuffle(self, data_type: str, batch_size: int = 512):
        """Iterate over pre-shuffled shards with behaviour-sequence features."""
        yield from self._iter_seq(self._files(data_type, shuffled=True), batch_size)


class AmazonLoader(SeqTFRecordLoader):
    """Loader for the Amazon (3-domain) benchmark."""

    def __init__(self, tfrecord_path: str):
        super().__init__(tfrecord_path, CONFIGS["amazon"])


class Antm2cLoader(SeqTFRecordLoader):
    """Loader for the AntM2C (5-domain) benchmark."""

    def __init__(self, tfrecord_path: str):
        super().__init__(tfrecord_path, CONFIGS["antm2c"])
