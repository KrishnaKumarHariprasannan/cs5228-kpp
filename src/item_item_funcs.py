from __future__ import absolute_import, division

from src.transformers import *


def get_normalized_cols_item_item(df_recommend, cols):
    df_to_be_normalized = df_recommend.loc[:, cols]
    max_ = df_to_be_normalized.max()
    min_ = df_to_be_normalized.min()
    # print(max_)
    # print(min_)
    df_normalized = (df_to_be_normalized - min_) / (max_ - min_)
    return df_normalized


def get_top_recommendations_ii(
    similarity_item_item, indices_ii_ordered, x_items_ii, y_items_ii, k, df
):
    similarity_item_item_ordered = similarity_item_item[indices_ii_ordered]
    x_items_ordered = x_items_ii[indices_ii_ordered]
    y_items_ordered = y_items_ii[indices_ii_ordered]
    x_top_k = x_items_ordered[:k]
    topk_indices = indices_ii_ordered[:k]
    df_result = df.iloc[topk_indices, :]

    return df_result
