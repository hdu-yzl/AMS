"""Amazon step 2: remap user/item ids to contiguous integers and split per user.

For every user the interactions are sorted chronologically and split into
train/val/test by an 80/10/10 ratio. Users with fewer than 3 interactions are
assigned entirely to the training set. The original ids are preserved in the
``original_user_id`` / ``original_item_id`` columns so that the raw metadata can
be joined later.

Input:
    merged_data/<merged>.csv  (output of step 1)
Output:
    split_data/{train,val,test}.csv
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd


def collect_feature_values(data, feature_columns):
    feature_values = defaultdict(set)
    for col in feature_columns:
        feature_values[col].update(data[col].dropna().unique())
    return feature_values


def create_feature_mapping(feature_values):
    """Map every feature value to a unique integer id (ids start at 1)."""
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


def split_user_data(user_data, splitting):
    """Chronologically split a single user's interactions into train/val/test."""
    if len(user_data) < 3:
        return user_data, pd.DataFrame(), pd.DataFrame()

    user_data = user_data.sort_values(by="timestamp")

    tot_ratio = sum(splitting)
    ratios = [r / tot_ratio for r in splitting if r > 0.0]
    split_ratios = np.cumsum(ratios)[:-1]
    split_indices = (len(user_data) * split_ratios).astype(int)

    train = user_data.iloc[: split_indices[0]]
    val = user_data.iloc[split_indices[0]: split_indices[1]]
    test = user_data.iloc[split_indices[1]:]
    return train, val, test


def main():
    merged_data_path = "merged_data/ratings_books_ratings_sports_and_outdoors_ratings_movies_and_tv.csv"
    data = pd.read_csv(merged_data_path, sep="\t", header=0)

    feature_columns = ["user_id", "item_id"]
    feature_values = collect_feature_values(data, feature_columns)

    data.loc[:, "original_item_id"] = data["item_id"]
    data.loc[:, "original_user_id"] = data["user_id"]

    feature_mapping = create_feature_mapping(feature_values)
    df = remap_features(data, feature_mapping, feature_columns)

    splitting = [0.8, 0.1, 0.1]
    save_path = "split_data"
    os.makedirs(save_path, exist_ok=True)

    all_train, all_val, all_test = [], [], []
    for domain in df["domain"].unique():
        domain_data = df[df["domain"] == domain].copy()
        for _user_id, user_data in domain_data.groupby("user_id"):
            train, val, test = split_user_data(user_data, splitting)
            all_train.append(train)
            all_val.append(val)
            all_test.append(test)
        print(f"Domain {domain} split done.")

    pd.concat(all_train).to_csv(f"{save_path}/train.csv", index=False)
    pd.concat(all_val).to_csv(f"{save_path}/val.csv", index=False)
    pd.concat(all_test).to_csv(f"{save_path}/test.csv", index=False)
    print(f"Saved splits to {save_path}.")


if __name__ == "__main__":
    main()
