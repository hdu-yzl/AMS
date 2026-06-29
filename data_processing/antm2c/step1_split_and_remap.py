"""AntM2C step 1: parse raw logs, remap ids and split by date.

User and item ids are remapped to contiguous integers using a global mapping
built from all parts. Samples are split chronologically into train/val/test by
log date. The original item id is kept in ``original_item_id`` for image lookup.

Input:
    antm2c_10m_part{0,1,2}      raw log files
Output:
    {train,val,test}.csv
"""

from collections import defaultdict

import numpy as np
import pandas as pd

FILE_PATHS = ["antm2c_10m_part0", "antm2c_10m_part1", "antm2c_10m_part2"]

COLUMNS = [
    "user_id", "item_id", "log_time", "label", "bill_entity_seq", "service_entity_seq",
    "query_entity_seq", "item_entity_names", "item_title", "scene",
] + [f"deep_features_{i}" for i in range(27)]

SAVE_COLUMNS = [
    "user_id", "item_id", "original_item_id", "log_time", "label", "bill_entity_seq",
    "service_entity_seq", "query_entity_seq", "item_entity_names", "item_title", "scene",
]


def collect_feature_values(data, feature_columns):
    feature_values = defaultdict(set)
    for col in feature_columns:
        feature_values[col].update(data[col].dropna().unique())
    return feature_values


def create_feature_mapping(feature_values):
    feature_mapping = {}
    current_mapping_value = 1
    for col, values in feature_values.items():
        feature_mapping[col] = {value: current_mapping_value + i for i, value in enumerate(values)}
        current_mapping_value += len(values)
    return feature_mapping


def remap_features(data, feature_mapping, feature_columns):
    for col in feature_columns:
        data.loc[:, col] = data[col].map(feature_mapping[col])
    return data


def main():
    np.random.seed(2025)

    # Build a global id mapping over all parts first.
    all_data = pd.concat(
        [pd.read_csv(fp, names=COLUMNS, header=None, low_memory=False)[1:] for fp in FILE_PATHS],
        ignore_index=True,
    )
    feature_columns = ["user_id", "item_id"]
    feature_mapping = create_feature_mapping(collect_feature_values(all_data, feature_columns))

    train_list, val_list, test_list = [], [], []
    for file_path in FILE_PATHS:
        data = pd.read_csv(file_path, names=COLUMNS, header=None, low_memory=False)[1:]
        data["log_time"] = pd.to_datetime(data["log_time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        data["log_time"] = data["log_time"].fillna(pd.Timestamp("2023-08-01 12:00:00"))

        train_data = data[data["log_time"] <= "2023-08-03"]
        val_data = data[(data["log_time"] > "2023-08-03") & (data["log_time"] <= "2023-08-05")]
        test_data = data[data["log_time"] > "2023-08-05"]

        for part in (train_data, val_data, test_data):
            part.loc[:, "original_item_id"] = part["item_id"]

        train_list.append(remap_features(train_data, feature_mapping, feature_columns))
        val_list.append(remap_features(val_data, feature_mapping, feature_columns))
        test_list.append(remap_features(test_data, feature_mapping, feature_columns))

    pd.concat(train_list, ignore_index=True)[SAVE_COLUMNS].to_csv("train.csv", index=False)
    pd.concat(val_list, ignore_index=True)[SAVE_COLUMNS].to_csv("val.csv", index=False)
    pd.concat(test_list, ignore_index=True)[SAVE_COLUMNS].to_csv("test.csv", index=False)
    print("Saved train/val/test.csv")


if __name__ == "__main__":
    main()
