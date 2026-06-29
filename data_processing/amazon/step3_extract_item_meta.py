"""Amazon step 3: collect per-item metadata from the raw meta files.

The Amazon metadata files (``meta_*.json.gz``) are joined against the remapped
item ids produced in step 2, deduplicated and saved as a single CSV that the
text feature extractor (step 4) consumes.

Input:
    split_data/{train,val,test}.csv  (for the original->mapped item id table)
    data/meta_*.json.gz
Output:
    split_data/item_meta.csv
"""

import gzip

import pandas as pd

FILE_PATHS = [
    "split_data/train.csv",
    "split_data/val.csv",
    "split_data/test.csv",
]

META_FILES = [
    "data/meta_Sports_and_Outdoors.json.gz",
    "data/meta_Movies_and_TV.json.gz",
    "data/meta_Books.json.gz",
]


def load_item_id_mapping():
    """Build a {original_item_id -> mapped item_id} dictionary from the splits."""
    frames = [pd.read_csv(fp, low_memory=False) for fp in FILE_PATHS]
    all_data = pd.concat(frames)
    return all_data.set_index("original_item_id")["item_id"].to_dict()


def parse(path):
    with gzip.open(path, "rb") as g:
        for line in g:
            # The raw meta files store python-literal dicts, one per line.
            yield eval(line)


def get_df(path):
    rows = {i: record for i, record in enumerate(parse(path))}
    return pd.DataFrame.from_dict(rows, orient="index")


def main():
    item_id_mapping = load_item_id_mapping()
    all_meta_df = pd.DataFrame()

    for meta_file in META_FILES:
        print(f"Processing {meta_file}...")
        meta_df = get_df(meta_file)
        meta_df["itemID"] = meta_df["asin"].map(item_id_mapping)
        meta_df.dropna(subset=["itemID"], inplace=True)
        meta_df["itemID"] = meta_df["itemID"].astype("int64")

        ori_cols = meta_df.columns.tolist()
        ret_cols = [ori_cols[-1]] + ori_cols[:-1]
        all_meta_df = pd.concat([all_meta_df, meta_df[ret_cols]], ignore_index=True)

    all_meta_df.sort_values(by=["itemID"], inplace=True)
    all_meta_df.drop_duplicates(subset=["itemID"], inplace=True)

    save_path = "split_data/item_meta.csv"
    all_meta_df.to_csv(save_path, index=False)
    print(f"Saved metadata to {save_path}, length: {len(all_meta_df)}")


if __name__ == "__main__":
    main()
