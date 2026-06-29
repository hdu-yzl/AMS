"""Amazon step 1: merge three category subsets and generate binary labels.

Each of the three rating files becomes one domain. Ratings greater than 3 are
treated as positive (clicked) samples. Only users that appear in all three
domains and have at least 5 interactions are kept.

Input:
    <data_path>/<category>.csv  with columns: user_id,item_id,rating,timestamp
Output:
    <save_path>/<a>_<b>_<c>.csv with columns: user_id,item_id,domain,label,timestamp
"""

import os

import pandas as pd


def concat_df(df1, df2, df3, filtered_column_name="user_id"):
    """Keep only users that appear in all three domains, then concatenate."""
    unique_user_1 = set(df1[filtered_column_name].unique())
    unique_user_2 = set(df2[filtered_column_name].unique())
    unique_user_3 = set(df3[filtered_column_name].unique())
    common_user = unique_user_1 & unique_user_2 & unique_user_3
    df1_filtered = df1[df1[filtered_column_name].isin(common_user)]
    df2_filtered = df2[df2[filtered_column_name].isin(common_user)]
    df3_filtered = df3[df3[filtered_column_name].isin(common_user)]
    return pd.concat([df1_filtered, df2_filtered, df3_filtered], ignore_index=True)


def generate_merged_dataset(data_path, save_path, data_a_name, data_b_name, data_c_name):
    pure_a_file = data_path + data_a_name + ".csv"
    pure_b_file = data_path + data_b_name + ".csv"
    pure_c_file = data_path + data_c_name + ".csv"

    names = ["user_id", "item_id", "rating", "timestamp"]
    df_a = pd.read_csv(pure_a_file, sep=",", header=None, names=names)
    df_b = pd.read_csv(pure_b_file, sep=",", header=None, names=names)
    df_c = pd.read_csv(pure_c_file, sep=",", header=None, names=names)

    # Assign one domain id per category.
    df_a["domain"] = 0
    df_b["domain"] = 1
    df_c["domain"] = 2

    # Rating > 3 -> positive label.
    for df in (df_a, df_b, df_c):
        df["label"] = df["rating"].apply(lambda x: 1 if x > 3 else 0)

    final_inter_data = concat_df(df_a, df_b, df_c)
    # Keep only users with at least 5 interactions overall.
    user_count = final_inter_data["user_id"].value_counts()
    final_inter_data = final_inter_data[final_inter_data["user_id"].isin(user_count[user_count >= 5].index)]

    print("Total interactions: {}".format(len(final_inter_data)))
    print("Total users: {}".format(len(final_inter_data["user_id"].unique())))
    print("Positive samples: {}".format(len(final_inter_data[final_inter_data["label"] == 1])))
    print("Negative samples: {}".format(len(final_inter_data[final_inter_data["label"] == 0])))
    for d in (0, 1, 2):
        n_items = len(final_inter_data[final_inter_data["domain"] == d]["item_id"].unique())
        print(f"Distinct items in domain {d}: {n_items}")

    os.makedirs(save_path, exist_ok=True)
    out_name = data_a_name.lower() + "_" + data_b_name.lower() + "_" + data_c_name.lower()
    final_inter_data[["user_id", "item_id", "domain", "label", "timestamp"]].to_csv(
        f"{save_path}/{out_name}.csv", index=False, sep="\t", header=True,
    )


if __name__ == "__main__":
    generate_merged_dataset(
        data_path="data/",
        data_a_name="ratings_Books",
        data_b_name="ratings_Sports_and_Outdoors",
        data_c_name="ratings_Movies_and_TV",
        save_path="merged_data/",
    )
