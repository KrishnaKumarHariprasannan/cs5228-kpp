import sys
sys.path.append("..")
import pandas as pd
import numpy as np
from src.transformers import *
from src.user_item_funcs import *
from src.item_item_funcs import *
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

train = pd.read_csv("../data/processed/train.csv", sep=",")
test = pd.read_csv("../data/processed/test.csv", sep=",")

def get_user_recoms(price, depreciation, type_of_vehicle, top_n):
    user_pref_dict = {'price':price, 'depreciation':depreciation, 'type_of_vehicle':type_of_vehicle}
    if (not price) or (price=='Select'):
        user_pref_dict.pop('price')
    if (not depreciation) or (depreciation=='Select'):
        user_pref_dict.pop('depreciation')
    if (not type_of_vehicle) or (type_of_vehicle=='Select'):
        user_pref_dict.pop('type_of_vehicle')
    # print(user_pref_dict)
    user_pref_cols = get_user_pref_cols(user_pref_dict)
    df_train_user_pref_cols = train.loc[:, ['listing_id']+user_pref_cols]
    df_prepared_num = prepare_df_num(df_train_user_pref_cols, user_pref_dict, pref_dict_num)
    user_row_dict = prepare_user_row_dict(df_prepared_num, user_pref_dict, pref_dict_num, pref_dict_cat)
    df_prepared_normalized = normalize_df(df_prepared_num, user_pref_cols, pref_dict_num)
    df_prepared_normalized = prepare_df_cat(df_prepared_normalized, user_pref_dict, pref_dict_cat)
    df_user_row, cols_to_be_dropped = get_user_row(user_row_dict, pref_dict_num, pref_dict_cat, df_prepared_normalized)
    df_prepared_final = df_prepared_normalized.drop(cols_to_be_dropped, axis=1)
    df_user_row_final = df_user_row.drop(cols_to_be_dropped, axis=1)

    x_user = df_user_row_final['listing_id'].to_numpy()
    y_user = df_user_row_final.iloc[:,1:].to_numpy()
    # print(x_user.shape,y_user.shape)
    x_items = df_prepared_final['listing_id'].to_numpy()
    y_items = df_prepared_final.iloc[:,1:].to_numpy()
    # print(x_items.shape, y_items.shape)
    similarity_user_item = cosine_similarity(y_user, y_items)[0]
    ind_ordered = np.argsort(similarity_user_item)[::-1]
    df_top_user_item = get_top_recommendations_user_item(similarity_user_item, ind_ordered, x_items, y_items, top_n, train)
    # print('user_pref_dict: ', user_pref_dict)
    # return df_top_user_item.loc[:, ['listing_id', 'make', 'model', 'depreciation', 'price', 'type_of_vehicle']]
    return df_top_user_item

def get_similar_items(listing_id_chosen, top_n=10):
    df_recommend_ii = train.loc[:, ['listing_id','make', 'vehicle_age', 'type_of_vehicle', 'depreciation',
                   'dereg_value', 'mileage', 'price', 'engine_cap',  'fuel_type_diesel',
                   'fuel_type_petrol-electric', 'fuel_type_petrol', 'fuel_type_electric','transmission_auto',
                   'transmission_manual', 'brand_rank']]

    cols_to_be_normalized = ['vehicle_age', 'depreciation', 'dereg_value', 'mileage', 'price']
    df_normalized_ii = get_normalized_cols_item_item(df_recommend_ii, cols_to_be_normalized)
    df_recommend_ii.loc[:, ['vehicle_age', 'depreciation', 'dereg_value', 'mileage', 'price']] = df_normalized_ii
    df_transformed_ii = pd.get_dummies(df_recommend_ii, columns = ['make', 'type_of_vehicle', 'brand_rank'])

    x_items_ii = df_transformed_ii['listing_id'].to_numpy()
    y_items_ii = df_transformed_ii.iloc[:,1:].to_numpy()
    x_chosen = listing_id_chosen
    y_chosen = df_transformed_ii[df_transformed_ii['listing_id'] == x_chosen].iloc[:,1:].to_numpy()

    similarity_item_item = cosine_similarity(y_chosen, y_items_ii)[0]
    indices_ii_ordered = np.argsort(similarity_item_item)[::-1]
    df_top_ii = get_top_recommendations_ii(similarity_item_item, indices_ii_ordered, x_items_ii, y_items_ii, top_n+1, train)
    df_top_ii = df_top_ii[df_top_ii.listing_id != listing_id_chosen]
    # df_top_ii = df_top_ii[df_top_ii.listing_id != listing_id_chosen]
    # return df_top_ii.loc[:, ['listing_id','make', 'model', 'vehicle_age', 'type_of_vehicle', 'depreciation',
    #                'dereg_value', 'mileage', 'price', 'engine_cap',  'fuel_type','transmission', 'brand_rank']]
    return df_top_ii
