import pandas as pd
import numpy as np
from src.transformers import *
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
import time

price_ranges = ['Below $10,000', '$10,001 to $20,000', '$20,001 to $30,000', '$30,001 to $40,000', '$40,001 to $50,000',
               '$50,001 to $60,000', '$60,001 to $70,000', '$70,001 to $80,000', '$80,001 to $100,000', '$100,001 to $120,000',
               '$120,001 to $140,000', '$140,001 to $160,000', '$160,001 to $180,000', '$180,001 to $200,000', 'Above $200,000']
price_bins = [0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 100000, 120000, 140000, 160000,
                 180000, 200000, np.inf]

depreciation_ranges = ['Below $10k /yr', '$10k to $11k /yr', '$11k to $12k /yr', '$12k to $13k /yr', '$13k to $14k /yr',
                      '$14k to $16k /yr', '$16k to $18k /yr', '$18k to $20k /yr', '$20k to $25k /yr', 'Above $25k /yr']
depreciation_bins = [0, 10000, 11000, 12000, 13000, 14000, 16000, 18000, 20000, 25000, np.inf]

type_of_vehicle_ranges = ['sports car', 'luxury sedan', 'suv', 'hatchback','mid-sized sedan', 'stationwagon',
                          'mpv', 'bus/mini bus', 'truck','others', 'van']
pref_dict_num = {'price':{"ranges":price_ranges, 'bins':price_bins}, 'depreciation':{"ranges":depreciation_ranges,
                                                                                    'bins':depreciation_bins}}
pref_dict_cat = {'type_of_vehicle': type_of_vehicle_ranges}

def get_user_pref_cols(user_pref_dict):
    cols = list(user_pref_dict.keys())
    return cols

def prepare_df_num(df_train_user_pref_cols, user_pref_dict, pref_dict_num):
    df = df_train_user_pref_cols.copy()
    for col, val_range in user_pref_dict.items():
        if col in pref_dict_num:
            pipe = Pipeline(
                [
                    (
                        col+'_range',
                        ColumnValuesToCategory(
                            col,
                            col+"_range",
                            pref_dict_num[col]['bins'],
                            pref_dict_num[col]['ranges'],
                        ),
                    )
                ]
            )

            df = pipe.fit_transform(df)

    return df

def prepare_df_cat(df_prepared_num, user_pref_dict, pref_dict_cat):
    df = df_prepared_num.copy()
    for col, val_range in user_pref_dict.items():
        if col in pref_dict_cat:
#             print('col: ', col)
            df = pd.get_dummies(df, columns = [col], prefix='', prefix_sep='')

    return df

def normalize_df(df_prepared_num, user_pref_cols, pref_dict_num):
    df_prep = df_prepared_num.copy()
    cols = []
    for col in user_pref_cols:
        if col in pref_dict_num:
            cols.append(col)
    df = df_prep.loc[:, cols]
    df_normalized = (df - df.min()) / (df.max() - df.min())
    df_prep.loc[:, cols] = df_normalized
    return df_prep

def get_col_value(val_range, pref_dict_num, col):
    ranges = pref_dict_num[col]['ranges']
    bins = pref_dict_num[col]['bins']
    index_ranges = ranges.index(val_range)
#     print('val_range: ', val_range)
#     print('ranges: ', ranges)
#     print('index_ranges: ', index_ranges)
    if index_ranges == len(bins)-2:
        value = 2*bins[index_ranges]-bins[index_ranges-1]
    else:
        value = (bins[index_ranges] + bins[index_ranges+1]) / 2
    return value

def prepare_user_row_dict(df_prepared_num, user_pref_dict, pref_dict_num, pref_dict_cat):
    row_dict ={}
    row_dict['listing_id'] = 0
    for col, val_range in user_pref_dict.items():
        if col in pref_dict_num:
            col_max = df_prepared_num.loc[:, col].max()
            col_min = df_prepared_num.loc[:, col].min()
            col_val = get_col_value(val_range, pref_dict_num, col)
            col_val_normalized = (col_val - col_min) / (col_max - col_min)
            row_dict[col] = [col_val_normalized]
        elif col in pref_dict_cat:
            row_dict[col] = [val_range]
#     print('row_dict: ', row_dict)
    return row_dict

def get_user_row(user_row_dict, pref_dict_num, pref_dict_cat, df_prepared_normalized):
    user_row ={}
    for col, val in user_row_dict.items():
        if col in pref_dict_num:
            user_row[col] = val[0]
        elif col in pref_dict_cat:
            for v in val:
                user_row[v] = 1
    # print(user_row)
    df_user_row = pd.DataFrame(columns = df_prepared_normalized.columns)
    # print(df_user_row.dtypes)
    for new_col in df_user_row.columns:
        if new_col not in user_row:
            user_row[new_col] = 0

    df_user_row = df_user_row.append(user_row,ignore_index=True)

    convert_dtype_dict = {}
    cols_to_be_dropped = []
    for col in df_user_row.columns:
        convert_dtype_dict[col] = str(df_prepared_normalized[col].dtype)
        if col in pref_dict_num:
            cols_to_be_dropped.append(col+'_range')
    # convert_dtype_dict['listing_id'] = 'int32'
    # print(convert_dtype_dict)
    df_user_row = df_user_row.astype(convert_dtype_dict)
    return df_user_row, cols_to_be_dropped

def get_top_recommendations_user_item(similarity_user_item, ind_ordered, x_items, y_items, k, df_prepared_final):
    similarity_user_item_ordered = similarity_user_item[ind_ordered]
    x_items_ordered = x_items[ind_ordered]
    y_items_ordered = y_items[ind_ordered]
    x_top_k = x_items_ordered[:k]
    count = 0
    per_num = 100
    while count < k:
        per_num = per_num - 1
        percentile = np.percentile(similarity_user_item, per_num)
        count = (similarity_user_item_ordered > percentile).sum()

    print('ind_ordered: ', ind_ordered)
    sample_similarity_user_item_ordered = similarity_user_item_ordered[:count]
    sample_indices = ind_ordered[:count]
    sample_prob = sample_similarity_user_item_ordered / np.sum(sample_similarity_user_item_ordered)
    topk_indices = np.random.choice(sample_indices, size=k, replace=False, p=sample_prob)
    print('topk_indices: ', topk_indices)
    df_result = df_prepared_final.iloc[topk_indices,:]

    return df_result
