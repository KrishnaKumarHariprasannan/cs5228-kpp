from __future__ import absolute_import, division

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from src.transformers import *

price_ranges = [
    "Below $10,000",
    "$10,001 to $20,000",
    "$20,001 to $30,000",
    "$30,001 to $40,000",
    "$40,001 to $50,000",
    "$50,001 to $60,000",
    "$60,001 to $70,000",
    "$70,001 to $80,000",
    "$80,001 to $100,000",
    "$100,001 to $120,000",
    "$120,001 to $140,000",
    "$140,001 to $160,000",
    "$160,001 to $180,000",
    "$180,001 to $200,000",
    "Above $200,000",
]
price_bins = [
    0,
    10000,
    20000,
    30000,
    40000,
    50000,
    60000,
    70000,
    80000,
    100000,
    120000,
    140000,
    160000,
    180000,
    200000,
    np.inf,
]

depreciation_ranges = [
    "Below $10k /yr",
    "$10k to $11k /yr",
    "$11k to $12k /yr",
    "$12k to $13k /yr",
    "$13k to $14k /yr",
    "$14k to $16k /yr",
    "$16k to $18k /yr",
    "$18k to $20k /yr",
    "$20k to $25k /yr",
    "Above $25k /yr",
]
depreciation_bins = [
    0,
    10000,
    11000,
    12000,
    13000,
    14000,
    16000,
    18000,
    20000,
    25000,
    np.inf,
]

type_of_vehicle_ranges = [
    "sports car",
    "luxury sedan",
    "suv",
    "hatchback",
    "mid-sized sedan",
    "stationwagon",
    "mpv",
    "bus/mini bus",
    "truck",
    "others",
    "van",
]
my_dict = {
    "price": {"ranges": price_ranges, "bins": price_bins},
    "depreciation": {"ranges": depreciation_ranges, "bins": depreciation_bins},
}


def prepare_df_ranges(df_train_user_pref_cols, user_pref_dict):
    df = df_train_user_pref_cols.copy()
    for col, val_range in user_pref_dict.items():
        if col in my_dict:
            pipe = Pipeline(
                [
                    (
                        col + "_range",
                        ColumnValuesToCategory(
                            col,
                            col + "_range",
                            my_dict[col]["bins"],
                            my_dict[col]["ranges"],
                        ),
                    )
                ]
            )

            df = pipe.fit_transform(df)
    return df


def get_user_pref_cols(user_pref_dict):
    cols = list(user_pref_dict.keys())
    return cols


def prepare_df_one_hot(df_to_be_one_hot, user_pref_dict, my_dict):
    df = df_to_be_one_hot.copy()
    for col, val_range in user_pref_dict.items():
        if col in my_dict:
            #             df = df.drop(col, axis=1)
            df = pd.get_dummies(
                df, columns=[col + "_range"], prefix=col, prefix_sep="_"
            )
        else:
            df = pd.get_dummies(df, columns=[col], prefix="", prefix_sep="")
    return df


def get_top_recommendations_user_item(
    similarity_user_item, ind_ordered, x_items, y_items, k, df_prepared_final
):
    similarity_user_item_ordered = similarity_user_item[ind_ordered]
    #     x_items_ordered = x_items[ind_ordered]
    #     y_items_ordered = y_items[ind_ordered]
    similarity_user_item_ordered = similarity_user_item[ind_ordered]
    topk_indices = ind_ordered[:k]
    df_result = df_prepared_final.iloc[topk_indices, :]
    # print('similarity_user_item_ordered: ', similarity_user_item_ordered)
    return df_result


def prepare_df_user_row(df_one_hot, my_dict, user_pref_dict):
    user_row = {}
    df_user_row = pd.DataFrame(columns=df_one_hot.columns)
    convert_dtype_dict = {}
    for col in df_one_hot.columns:
        convert_dtype_dict[col] = str(df_one_hot[col].dtype)
        if col in my_dict:
            user_row[col + "_" + user_pref_dict[col]] = 1
        elif col in user_pref_dict.values():
            user_row[col] = 1
    df_user_row = df_user_row.append(user_row, ignore_index=True)
    df_user_row = df_user_row.fillna(0)
    df_user_row = df_user_row.astype(convert_dtype_dict)
    return df_user_row


# def get_top_recommendations_user_item_with_sampling(similarity_user_item, ind_ordered, x_items, y_items, k, df_prepared_final):
#     similarity_user_item_ordered = similarity_user_item[ind_ordered]
#     x_items_ordered = x_items[ind_ordered]
#     y_items_ordered = y_items[ind_ordered]
#     similarity_user_item_ordered = similarity_user_item[ind_ordered]
#     x_top_k = x_items_ordered[:k]
#     count = 0
#     per_num = 100
# #     print('np.percentile(similarity_user_item, per_num): ', np.percentile(similarity_user_item, per_num))
#     while count < k:
#         per_num = per_num - 1
#         percentile = np.percentile(similarity_user_item, per_num)
#         count = (similarity_user_item_ordered > percentile).sum()
#
#     print('count: ', count)
#     print('ind_ordered: ', ind_ordered)
# #     print('similarity_user_item_ordered[:count]: ', similarity_user_item_ordered[:count])
#
#     sample_similarity_user_item_ordered = similarity_user_item_ordered[:count]
#     sample_indices = ind_ordered[:count]
#     sample_prob = sample_similarity_user_item_ordered / np.sum(sample_similarity_user_item_ordered)
# #     print('sample_prob: ', sample_prob)
#     topk_indices = np.random.choice(sample_indices, size=k, replace=False, p=sample_prob)
#     print('topk_indices: ', topk_indices)
#     df_result = df_prepared_final.iloc[topk_indices,:]
# #     df_result = df_prepared_final.iloc[ind_ordered[:k],:]
#
#     return df_result
