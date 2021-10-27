from __future__ import absolute_import, division

import logging
import numpy as np
import pandas as pd
import re
import scipy

from datetime import datetime
from pandas.api.types import CategoricalDtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


# By observation of "coe_rebate", "dereg_value", "dereg_value_computed" for a few samples
DATASET_GENERATION_DATE = datetime(2021, 9, 14)
# Date on which we scraped coe text data from the web portal
COE_SCRAPE_DATE = datetime(2021, 10, 26)
FIELDS_TO_DROP = ["indicative_price", "eco_category"]


def get_make_from_title(make_list, title):
    title = title.split(" ")
    for i in range(len(title)):
        if " ".join(title[0: i + 1]) in make_list:
            return " ".join(title[0: i + 1])
    return "unknwon"


def make_category_vector(cat_list, x):
    vector = [0] * len(cat_list)
    for i, cat in enumerate(cat_list):
        if cat in x:
            vector[i] = 1
    return vector


class PreProcessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.make_list = []
        pass

    def fit(self, df):
        self.make_list = df.make.unique()
        return self

    def transform(self, df):
        df.loc[:, "reg_date"] = np.where(
            df["reg_date"].isnull(), df["original_reg_date"], df["reg_date"]
        )
        df.loc[:, "reg_date"] = pd.to_datetime(df.reg_date)
        df.loc[:, "reg_date_year"] = pd.to_datetime(df.reg_date).dt.year

        df.loc[:, "manufactured"] = np.where(
            (df.manufactured.isnull()) | (df.manufactured > 2021),
            df["reg_date_year"],
            df["manufactured"],
        )

        df.loc[:, "no_of_owners"] = df["no_of_owners"].fillna(1)
        df.loc[:, "title"] = df["title"].str.lower()
        df.loc[:, "make"] = df.apply(
            lambda row: get_make_from_title(self.make_list, row["title"])
            if pd.isnull(row["make"])
            else row["make"],
            axis=1,
        )
        df.loc[:, "make_model"] = df.make + "-" + "df.model"
        return df


class PostProcessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns_to_drop = [
            "indicative_price",
            "original_reg_date",
            "opc_scheme",
            "lifespan",
            "fuel_type",
            "description",
            "features",
            "accessories",
            "listing_id",
            "title",
            "eco_category",
            "reg_date",
            "road_tax",
            "model",
            "category",
            "make",
            "price",
        ]
        pass

    def fit(self, df):
        #         self.make_list = df.make.unique()
        return self

    def transform(self, input_df):
        df = input_df.copy()
        df = df.drop(self.columns_to_drop, axis=1, errors="ignore")
        return df


class GroupMissingValueImputer(BaseEstimator, TransformerMixin):
    def __init__(self, col, group_cols, agg="mean"):
        self.group_mapping = {}
        self.group_cols = group_cols
        self.agg = agg
        self.col = col

    def fit(self, df):
        col = self.col
        if self.agg == "first":
            self.group_mapping = (
                df[~df[col].isnull()].groupby(
                    self.group_cols).first()[col].to_dict()
            )
        elif self.agg == "mean":
            self.group_mapping = (
                df[~df[col].isnull()].groupby(
                    self.group_cols).mean()[col].to_dict()
            )
        elif self.agg == "median":
            self.group_mapping = (
                df[~df[col].isnull()].groupby(
                    self.group_cols).median()[col].to_dict()
            )
        else:
            raise Exception("Unknown Agg type")
        return self

    def transform(self, input_df):
        col = self.col
        df = input_df.copy()
        if col is not None and col in df.columns:
            key = tuple(self.group_cols)
            if df[col].dtype == np.object_:
                unknown_value = "unknown"
            else:
                if self.agg == "mean":
                    unknown_value = df[col].mean()
                else:
                    unknown_value = df[col].median()
            result = df.apply(
                lambda row: self.group_mapping.get(key, unknown_value)
                if pd.isnull(row[col])
                else row[col],
                axis=1,
            )
            df.loc[:, col] = result
            return df
        return df


class MeanMissingValueImputer(BaseEstimator, TransformerMixin):
    def __init__(self, cols, agg="mean"):
        self.mapping = {}
        self.agg = agg
        self.cols = cols

    def fit(self, df):
        cols = self.cols
        for col in cols:
            self.mapping[col] = df[col].mean()
        return self

    def transform(self, input_df):
        cols = self.cols
        df = input_df.copy()
        for col in cols:
            df.loc[:, col] = df.fillna(self.mapping[col])
        return df


class SplitValuesToColumn(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        self.val_list = {}
        self.col = col

    def fit(self, df):
        categories = set()
        for values in df[self.col].unique():
            for value in values.split(","):
                if len(value.strip()) > 2:
                    categories.add(value.strip())

        self.val_list = categories
        return self

    def transform(self, input_df):
        col = self.col
        df = input_df.copy()
        df.reset_index(inplace=True, drop=True)
        df_cat = pd.DataFrame(
            df[col]
            .apply(
                lambda x: make_category_vector(
                    self.val_list, list(map(str.strip, x.split(",")))
                )
            )
            .tolist(), columns=self.val_list
        )
        df_cat = df_cat.add_prefix(col + "_")
        result = pd.concat([df, df_cat], axis=1)
        return result


class CarSpecificationsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, col, group_cols, agg="mean"):
        self.group_mapping_list = []
        self.group_cols = group_cols
        self.col = col
        self.agg = agg

    def get_key(self, row, group_columns):
        lst = []
        if len(group_columns) > 1:
            for c in group_columns:
                lst.append(row[c])
            return tuple(lst)
        else:
            return row[group_columns[0]]

    def fit(self, df):
        group_cols = self.group_cols.copy()
        col = self.col
        for i in range(len(group_cols)):
            if self.agg == "mean":
                group_mapping = (
                    df[~df[col].isnull()].groupby(
                        group_cols).mean()[col].to_dict()
                )
            elif self.agg == "mode":
                group_mapping = (
                    df[~df[col].isnull()]
                    .groupby(group_cols)
                    .agg(lambda x: scipy.stats.mode(x)[0])[col]
                    .to_dict()
                )
            self.group_mapping_list.append(group_mapping)
            group_cols.pop()
        return self

    def transform(self, input_df):
        group_cols = self.group_cols.copy()
        df = input_df.copy()
        for group_mapping in self.group_mapping_list:
            col = self.col

            if col is not None and col in df.columns:
                result = df.apply(
                    lambda row: group_mapping.get(
                        self.get_key(row, group_cols))
                    if pd.isnull(row[col])
                    else row[col],
                    axis=1,
                )
                df.loc[:, col] = result
            group_cols.pop()
        return df


class CarSpecsMissingWithTypeOfVehicle(BaseEstimator, TransformerMixin):
    def __init__(self, cols, agg="mean"):
        self.group_mapping = {}
        self.group_mapping_list = []
        self.cols = cols
        self.agg = agg

    def fit(self, df):
        for col in self.cols:
            if self.agg == "mean":
                group_mapping = (
                    df[~df[col].isnull()]
                    .groupby("type_of_vehicle")
                    .mean()[col]
                    .to_dict()
                )
            elif self.agg == "mode":
                group_mapping = (
                    df[~df[col].isnull()]
                    .groupby("type_of_vehicle")
                    .agg(lambda x: scipy.stats.mode(x)[0])[col]
                    .to_dict()
                )
            self.group_mapping_list.append(group_mapping)
        return self

    def transform(self, input_df):
        cols = self.cols
        df = input_df.copy()
        group_mapping_list = self.group_mapping_list

        for i in range(len(group_mapping_list)):
            col = cols[i]
            if col is not None and col in df.columns:
                result = df.apply(
                    lambda row: group_mapping_list[i].get(
                        row["type_of_vehicle"])
                    if pd.isnull(row[col])
                    else row[col],
                    axis=1,
                )
                df.loc[:, col] = result
        return df


class ColumnValuesToCategory(BaseEstimator, TransformerMixin):
    """
    Creates a new column (new_col) with the values of column (col) converted into categories.
    bins -> The ranges of the categories we want in the new column.
    names -> names of the ranges of the category that will be displayed in the new column (new_col).
    """

    def __init__(self, col, new_col, bins, names):
        self.bins = bins
        self.names = names
        self.col = col
        self.new_col = new_col

    def fit(self, df):
        return self

    def transform(self, input_df):
        df = input_df.copy()
        df[self.new_col] = pd.cut(df[self.col], self.bins, labels=self.names)
        return df


class OneHotTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, col, categories):
        self.col = col
        self.categories = categories

    def fit(self, df):
        return self

    def transform(self, input_df):
        df = input_df.copy()
        df.reset_index(inplace=True, drop=True)
        df[self.col] = df[self.col].astype(CategoricalDtype(self.categories))
        df = pd.concat(
            [df, pd.get_dummies(df[self.col], prefix=self.col)], axis=1)
        # can include code if we want to drop the column we one-hot encoded.
        # It has not been dropped yet as some other transformer might be using it.
        # Drop it later.
        # if self.col in df.columns:
        #     df.drop([self.col],axis=1, inplace=True)
        return df


class CoeTransformer(BaseEstimator, TransformerMixin):
    """
    Imputes missing coe values as well as incorrect ones
    """

    def fit(self, X):
        years_with_missing_coe = X[X["coe"].isnull()].coe_start_year.unique()
        res_of_interest = X[X["coe_start_year"].isin(years_with_missing_coe)]
        self.mean_coe_per_year = res_of_interest.groupby("coe_start_year").agg(
            {"coe": np.mean}
        )
        self.mean_coe = X["coe"].agg("mean")
        return self

    def transform(self, X):
        # Fill missing coe values with mean coe for that registration year
        combined_x = X.join(
            self.mean_coe_per_year, on="coe_start_year", rsuffix="_mean"
        )
        combined_x["coe"].fillna(combined_x["coe_mean"], inplace=True)
        combined_x.drop("coe_mean", axis=1, inplace=True)

        # Replace incorrect coe values with mean coe for 2021
        # Example: https://www.sgcarmart.com/used_cars/info.php?ID=1017335
        combined_x.coe.replace(
            10.0, self.mean_coe_per_year.loc[2021].coe, inplace=True)

        coe_mask = combined_x["coe"].isnull()

        if len(combined_x[coe_mask]):
            combined_x.loc[coe_mask, "coe"] = self.mean_coe

        return combined_x


class ArfTransformer(BaseEstimator, TransformerMixin):
    """
    Imputes missing arf values based on its corresponding omv
    """

    DEDUCTION_AMOUNT_TO_RATE_TUPLE = [(20000, 1), (30000, 1.4), (0, 1.8)]

    @classmethod
    def compute_arf(cls, omv):
        """
        Given an omv, compute its corresponding ARF per https://www.sgcarmart.com/news/writeup.php?AID=13
        """
        arf = 0
        for amount_to_deduct, rate in cls.DEDUCTION_AMOUNT_TO_RATE_TUPLE:
            if omv >= amount_to_deduct and amount_to_deduct != 0:
                arf += rate * amount_to_deduct
            else:
                arf += rate * omv

            omv -= amount_to_deduct

            if omv <= 0:
                break

        return arf

    def fit(self, X):

        return self

    def transform(self, df):
        X = df.copy()
        X.reset_index(inplace=True, drop=True)
        rows_without_arf = X[X["arf"].isnull()]
        computed_arf = rows_without_arf["omv"].apply(self.compute_arf)
        computed_arf.rename("arf_computed", inplace=True)
        modified_x = X.join(computed_arf)
        modified_x["arf"].fillna(modified_x["arf_computed"], inplace=True)
        modified_x.drop("arf_computed", axis=1, inplace=True)

        if len(modified_x[modified_x.arf.isnull()]):
            logging.info(
                f"ArfTransformer - found {len(modified_x[modified_x.arf.isnull()])} rows with null arf"
            )
            modified_x = modified_x[~modified_x.arf.isnull()]

        return modified_x


class AgeFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Adds a new column "vehicle_age" as min("manufactured", "reg_date_year")
    """

    def fit(self, X):

        return self

    def transform(self, X):
        modified_x = X.copy()
        modified_x["vehicle_age"] = datetime.now().year - np.min(
            X[["manufactured", "reg_date_year"]], axis=1
        )
        return modified_x


class ParfFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Adds a new column "parf" based on "vehicle_age" and "arf"
    """

    @classmethod
    def compute_parf(cls, row):
        """
        Compute parf from vehicle age and arf per https://www.sgcarmart.com/news/writeup.php?AID=13
        """
        parf = 0
        if not row["is_parf_car"]:
            return parf

        if row["vehicle_age"] <= 4:
            parf = row["arf"] * 0.75
        elif row["vehicle_age"] >= 5 and row["vehicle_age"] < 6:
            parf = row["arf"] * 0.70
        elif row["vehicle_age"] >= 6 and row["vehicle_age"] < 7:
            parf = row["arf"] * 0.65
        elif row["vehicle_age"] >= 7 and row["vehicle_age"] < 8:
            parf = row["arf"] * 0.60
        elif row["vehicle_age"] >= 8 and row["vehicle_age"] < 9:
            parf = row["arf"] * 0.55
        elif row["vehicle_age"] >= 9 and row["vehicle_age"] <= 10:
            parf = row["arf"] * 0.50

        return parf

    def fit(self, X):

        return self

    def transform(self, X):
        modified_x = X.copy()
        # If the car category does not contain "parf car" then it does not get any part rebate
        modified_x["is_parf_car"] = modified_x.category.apply(
            lambda value: 1 if "parf car" in value else 0
        )
        modified_x["parf"] = modified_x.apply(self.compute_parf, axis=1)
        return modified_x


class CoeStartDateFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Adds a new column coe_start_date based on the scraped coe text information
    """

    @classmethod
    def compute_coe_from_text(cls, value):
        delta = np.timedelta64(0, "M")

        if (type(value) != str) or "COE" not in value:
            return delta

        for unit_pattern_tuple in [
            ("Y", "([0-9]+)yr"),
            ("M", "([0-9]+)mth"),
            ("D", "([0-9]+)day"),
        ]:
            matches = re.findall(unit_pattern_tuple[1], value)
            if len(matches):
                # Convert day units to month before addding to delta
                # https://numpy.org/doc/stable/reference/arrays.datetime.html#datetime-and-timedelta-arithmetic
                if unit_pattern_tuple[0] == "D":
                    delta += np.timedelta64(
                        int(np.ceil(int(matches[0]) / 30)), "M")
                else:
                    delta += np.timedelta64(int(matches[0]),
                                            unit_pattern_tuple[0])
            if len(matches) > 1:
                raise ValueError(
                    f"Incorrect pattern identified for the combination - {listing_id} and {value}"
                )

        return delta

    def fit(self, X):
        return self

    def transform(self, input_df):
        print("modified")
        X = input_df.copy()
        X.reset_index(inplace=True, drop=True)
        
        if len(X) > 6000:
            file_path = "../data/raw/train_coe_text.csv"
        else:
            file_path = "../data/raw/test_coe_text.csv"

        coe_text = pd.read_csv(file_path)
        coe_text.set_index("listing_id", inplace=True)
        coe_text = coe_text[~coe_text.index.duplicated(keep="first")]

        merged_df = X.merge(coe_text, on="listing_id", how="left")

        merged_df["coe_expiry_days"] = merged_df.coe_text.apply(
            self.compute_coe_from_text
        )

        incorrect_coe_expiry_mask = merged_df["coe_expiry_days"] == np.timedelta64(
            0)

        merged_df["coe_expiry_months"] = merged_df["coe_expiry_days"] / np.timedelta64(
            1, "M"
        )
        merged_df["coe_expiry_date"] = COE_SCRAPE_DATE + \
            merged_df["coe_expiry_days"]
        merged_df["coe_start_date"] = merged_df["coe_expiry_date"] - np.timedelta64(
            10, "Y"
        )

        # 668 rows have inactive entries on the portal and so their coe_text entries are empty. For such
        # entries, compute the coe_start_date using known methods

        # Some rows have coe values as 10 - https://www.sgcarmart.com/used_cars/info.php?ID=1027957 (scraping error)
        # In such cases, consider DATASET_GENERATION_DATE as coe_start_date
        logging.info(
            f"CoeStartDateFeatureCreator - Found {len(merged_df[incorrect_coe_expiry_mask])} entries without COE Text"
        )

        coe_10_mask = incorrect_coe_expiry_mask & (merged_df["coe"] == 10)
        merged_df.loc[coe_10_mask, "coe_start_date"] = DATASET_GENERATION_DATE
        merged_df.loc[coe_10_mask, "coe_expiry_date"] = merged_df.loc[
            coe_10_mask, "coe_start_date"
        ] + np.timedelta64(10, "Y")
        merged_df.loc[coe_10_mask, "coe_expiry_months"] = (
            merged_df.loc[coe_10_mask, "coe_expiry_date"] -
            DATASET_GENERATION_DATE
        ) / np.timedelta64(1, "M")

        incorrect_coe_expiry_mask = merged_df.coe_expiry_months == 0
        logging.info(
            f"CoeStartDateFeatureCreator - {len(merged_df[incorrect_coe_expiry_mask])} null entries left after using COE 10 "
        )
        merged_df.loc[incorrect_coe_expiry_mask, "coe_start_date"] = merged_df.loc[
            incorrect_coe_expiry_mask, "reg_date"
        ]
        merged_df.loc[incorrect_coe_expiry_mask, "coe_expiry_date"] = merged_df.loc[
            incorrect_coe_expiry_mask, "coe_start_date"
        ] + np.timedelta64(10, "Y")
        merged_df.loc[incorrect_coe_expiry_mask, "coe_expiry_months"] = (
            merged_df.loc[incorrect_coe_expiry_mask, "coe_expiry_date"]
            - DATASET_GENERATION_DATE
        ) / np.timedelta64(1, "M")

        merged_df.coe_expiry_months.clip(lower=0, inplace=True)

        incorrect_coe_expiry_mask = merged_df.coe_expiry_months == 0
        logging.info(
            f"CoeStartDateFeatureCreator - {len(merged_df[incorrect_coe_expiry_mask])} null entries left after using reg_date "
        )

        # Out of 133 applicable entries, 8 do not fall under the assumption here which is that all remaining cars are COE cars
        merged_df.loc[incorrect_coe_expiry_mask, "coe_expiry_months"] = (
            merged_df[incorrect_coe_expiry_mask].dereg_value * 120
        ) / merged_df[incorrect_coe_expiry_mask].coe

        # If the dereg_value is NaN, assume that there are 120 months left on its COE
        # Example: https://www.sgcarmart.com/used_cars/info.php?ID=994090
        # 11 such rows are present here
        merged_df.loc[merged_df.coe_expiry_months.isnull(),
                      "coe_expiry_months"] = 120

        merged_df.loc[incorrect_coe_expiry_mask, "coe_expiry_date"] = (
            merged_df.loc[incorrect_coe_expiry_mask, "coe_expiry_months"].apply(
                lambda value: np.timedelta64(int(value), "M")
            )
            + DATASET_GENERATION_DATE
        )
        merged_df.loc[incorrect_coe_expiry_mask, "coe_start_date"] = merged_df.loc[
            incorrect_coe_expiry_mask, "coe_expiry_date"
        ] - np.timedelta64(10, "Y")

        merged_df["coe_start_year"] = merged_df["coe_start_date"].dt.year

        return merged_df


class CoeRebateFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Adds a new column "coe_rebate" based on "parf", "coe", and "reg_date"
    """

    def fit(self, X):
        return self

    def transform(self, X):
        modified_x = X.copy()
        modified_x["coe_rebate"] = (
            modified_x.coe * modified_x.coe_expiry_months) / 120

        # If the computed coe_rebate is 0 (those records for which the coe_start_date is incorrect),
        # use dereg_value as coe_rebate.
        # r.loc[(r.coe_rebate == 0) & (r.dereg_value != 0)]
        # All of these records are for cars that are older than 10 years (no ARF)
        # and so this should be completely safe
        #
        # NOTE: One exception is https://www.sgcarmart.com/used_cars/info.php?ID=1029135 where the
        # coe > coe_rebate - this needs further investigation
        modified_x["coe_rebate"] = np.where(
            (modified_x["coe_rebate"] == 0) & (modified_x["dereg_value"] != 0),
            modified_x["dereg_value"],
            modified_x["coe_rebate"],
        )

        return modified_x


class DeregValueComputedFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Adds a column dereg_value_computed based on coe_rebate and parf
    """

    def fit(self, X):
        return self

    def transform(self, X):
        modified_x = X.copy()
        modified_x["dereg_value_computed"] = X["coe_rebate"] + X["parf"]
        return modified_x


class DeregValueTransformer(BaseEstimator, TransformerMixin):
    """
    Imputes missing dereg_value values based on its corresponding dereg_value_computed
    """

    def __init__(self, fill_zero=True):
        super(DeregValueTransformer, self).__init__()
        self.fill_zero = fill_zero

    def fit(self, X):
        return self

    def transform(self, X):
        modified_x = X.copy()
        modified_x["dereg_value"] = np.where(
            X["dereg_value"].isnull(), X["dereg_value_computed"], X["dereg_value"],
        )

        # Drop rows/Fill with zero for which it is not possible to compute dereg_value
        # This amounts to 228 rows. Example: https://www.sgcarmart.com/used_cars/info.php?ID=1031249
        # The above is due to error in scraping script which outside the scope of our problem
        null_mask = modified_x["dereg_value"].isnull()
        if len(modified_x[null_mask]):
            if self.fill_zero:
                logging.info(
                    f"DeregValueTransformer - replacing {len(modified_x[null_mask])} null values with 0"
                )
                modified_x.loc[null_mask, "dereg_value"] = 0
            else:
                logging.info(
                    f"DeregValueTransformer - removing {len(modified_x[null_mask])} for which dereg_value cannot be computed"
                )
                modified_x = modified_x.loc[~null_mask]

        return modified_x


class DepreciationTransformer(BaseEstimator, TransformerMixin):
    """
    Imputes missing depreciation values based on its corresponding price and parf
    """

    def __init__(self, fill_zero=True):
        super(DepreciationTransformer, self).__init__()
        self.fill_zero = fill_zero

    def fit(self, X):
        return self

    def transform(self, X):
        modified_x = X.copy()
        depreciation_mask = X.depreciation.isnull()

        # Ideally, this should be (price - parf) / no_of_coe_years_left but this formula gives
        # depreciation which are vastly different to the ones in the given dataset - because of incorrect coe_start_date
        # which in turn is due to a scraping error in dataset generation
        #         modified_x.loc[depreciation_mask, "depreciation"] = (
        #             X.loc[depreciation_mask, "price"] - X.loc[depreciation_mask, "parf"]
        #         ) / 10
        if len(modified_x[depreciation_mask]):
            if self.fill_zero:
                logging.info(
                    f"DepreciationTransformer - replacing {len(modified_x[depreciation_mask])} null values with 0"
                )
                modified_x.loc[depreciation_mask, "depreciation"] = 0
            else:
                logging.info(
                    f"DepreciationTransformer - removing {len(modified_x[depreciation_mask])} rows with null depreciation"
                )
                modified_x = modified_x[~depreciation_mask]

        return modified_x


class OpcSchemeTransformer(BaseEstimator, TransformerMixin):
    """
    Standardizes and imputes opc_scheme values
    """

    REVISED_OPC = "revised_opc"
    NORMAL_OPC = "normal_opc"
    OLD_OPC = "old_opc"

    def fit(self, X):
        return self

    def transform(self, X):
        modified_x = X.copy()
        opc_scheme = modified_x["opc_scheme"]

        opc_scheme.replace(
            "revised opc scheme . learn more about opc schemes.",
            self.REVISED_OPC,
            inplace=True,
        )

        # Only record with this value - https://www.sgcarmart.com/used_cars/info.php?ID=989043
        opc_scheme.replace("1100", self.REVISED_OPC, inplace=True)

        opc_scheme.fillna(self.NORMAL_OPC, inplace=True)

        opc_scheme.replace(
            "old opc scheme . learn more about opc schemes.", self.OLD_OPC, inplace=True
        )
        modified_x["opc_scheme"] = opc_scheme
        return modified_x


class LifespanRestrictionFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Adds a new feature lifespan_restriction based on lifespan value
    1  - no restruction
    -1 - restriction applies
    """

    def fit(self, X):
        return self

    def transform(self, X):
        modified_x = X.copy()
        modified_x["lifespan_restriction"] = -1
        modified_x.loc[modified_x["lifespan"].isnull(),
                       "lifespan_restriction"] = 1

        return modified_x


class CountUniqueItemsFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Creates a new feature column that reflects the number of unique items in a
    string column that is separated by given separator
    """

    def __init__(self, feature, new_feature_name, separator=","):
        super(CountUniqueItemsFeatureCreator, self).__init__()
        self.feature = feature
        self.new_feature_name = new_feature_name
        self.separator = separator

    def fit(self, X):
        return self

    def transform(self, X):
        modified_x = X.copy()
        new_feature = pd.Series(
            np.zeros(len(X)), index=X.index, dtype=np.int16)
        new_feature.loc[~X[self.feature].isnull()] = X[~X[self.feature].isnull()][
            self.feature
        ].apply(lambda value: len(value.split(self.separator)))
        modified_x[self.new_feature_name] = new_feature
        return modified_x


class HierarchicalGroupImputer(BaseEstimator, TransformerMixin):
    """
    For missing values in the given feature, this imputer tries filling such values with the agg value
    derived from each group and as fallback uses the entire feature columns' agg value

    If fallback is True, for records that cannot be filled with agg value of any of the groups provided,
    it will be filled with the agg value of the feature column
    """

    RSUFFIX = "_computed"

    def __init__(self, feature, groups, agg_type, fallback=True):
        super(HierarchicalGroupImputer, self).__init__()
        self.feature = feature
        self.groups = groups
        # Order groups from most-specific to least-specific
        self.groups.sort(key=len, reverse=True)
        self.agg_type = agg_type
        self.fallback = fallback

    def fit(self, X):
        self.agg_results = {}
        for group in self.groups:
            self.agg_results[tuple(group)] = X.groupby(group).agg(
                {self.feature: self.agg_type}
            )

        # If there are still empty values after the above, fill those values with the
        # column-level agg value
        if self.fallback:
            self.feature_agg = X[self.feature].agg(self.agg_type)

        return self

    def transform(self, X):

        # If there are no empty records, return
        if not len(X[X[self.feature].isnull()]):
            logging.info(
                f"HierarchicalGroupImputer - found no null values to impute for {self.feature}"
            )
            return X

        modified_x = X.copy()
        logging.info(
            f"HierarchicalGroupImputer - total {(len(modified_x[modified_x[self.feature].isnull()]))} null values to impute for {self.feature}"
        )
        for group in self.groups:
            feature_computed = modified_x.join(
                self.agg_results[tuple(group)], on=group, rsuffix=self.RSUFFIX
            )[self.feature + self.RSUFFIX]
            modified_x[self.feature] = np.where(
                modified_x[self.feature].isnull(),
                feature_computed,
                modified_x[self.feature],
            )

            logging.info(
                f"HierarchicalGroupImputer - {(len(modified_x[modified_x[self.feature].isnull()]))} null values left for {self.feature} after imputing with group {group}"
            )
            if not len(modified_x[modified_x[self.feature].isnull()]):
                break

        # If there are still empty values after the above, fill those values with the
        # column-level agg value
        if len(modified_x[modified_x[self.feature].isnull()][self.feature]):

            if self.fallback:
                modified_x.loc[
                    modified_x[self.feature].isnull(), self.feature
                ] = self.feature_agg
            else:
                null_records = modified_x[self.feature].isnull()

                logging.info(
                    f"HierarchicalGroupImputer - removing {len(modified_x[null_records])} rows with null {self.feature} values"
                )

                modified_x = modified_x[~null_records]

        return modified_x


class BrandRankTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.brand_rank_mapping = {}
        pass

    def fit(self, df):
        temp_df = df.groupby("make").median("price")["price"].reset_index()
        thresholds = temp_df.price.quantile(
            [0.25, 0.50, 0.70, 0.80, 0.90, 1]).values
        rank = 0
        t0 = 0
        for t1 in thresholds:
            rank = rank + 1
            temp = temp_df[(temp_df.price <= t1) & (temp_df.price > t0)]
            t0 = t1
            values = temp.make.values
            for v in values:
                self.brand_rank_mapping[v] = rank

        return self

    def transform(self, input_df):
        df = input_df.copy()
        df.reset_index(inplace=True, drop=True)
        df.loc[:, "brand_rank"] = df.make.apply(
            lambda row: self.brand_rank_mapping.get(row, 0)
        )
        return df


class OheCategorical(BaseEstimator, TransformerMixin):
    def __init__(self, cols, drop_first=False):
        self.cols = cols
        self.drop_first = drop_first
        self.mapping = {}

    def fit(self, df):
        cols = self.cols
        for col in cols:
            enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
            self.mapping[col] = enc.fit(df[[col]])
        return self

    def transform(self, input_df):
        cols = self.cols
        df = input_df.copy()
        for col in cols:
            temp = self.mapping[col].transform(df[[col]])
            temp_df = pd.DataFrame(temp, columns=self.mapping[col].categories_)
            temp_df = temp_df.add_suffix(f"_{col}")
            df = pd.concat([df, temp_df], axis=1)
            df = df.drop([col], axis=1)
        return df
