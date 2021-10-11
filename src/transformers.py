
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import loguniform
from datetime import datetime
from datetime import datetime
import logging
import numpy as np
import pandas as pd
from sklearn import compose
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

from utils import *

from sklearn.metrics import mean_squared_error
from datetime import datetime
import logging
import numpy as np
import pandas as pd
from sklearn import compose
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import scipy

from sklearn.base import BaseEstimator, TransformerMixin


# By observation of "coe_rebate", "dereg_value", "dereg_value_computed" for a few samples
DATASET_GENERATION_DATE = datetime(2021, 9, 14)
FIELDS_TO_DROP = ["indicative_price", "eco_category"]



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
        df.loc[:, "reg_date_month"] = (
            datetime.now() - pd.to_datetime(df.reg_date)
        ) / np.timedelta64(1, "M")
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
        ]
        pass

    def fit(self, df):
        #         self.make_list = df.make.unique()
        return self

    def transform(self, input_df):
        df = input_df.copy()
        df = df.drop(self.columns_to_drop, axis=1, errors="ignore")
        return df

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.base import BaseEstimator, TransformerMixin


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
                df[~df[col].isnull()].groupby(self.group_cols).first()[col].to_dict()
            )
        elif self.agg == "mean":
            self.group_mapping = (
                df[~df[col].isnull()].groupby(self.group_cols).mean()[col].to_dict()
            )
        elif self.agg == "median":
            self.group_mapping = (
                df[~df[col].isnull()].groupby(self.group_cols).median()[col].to_dict()
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
            .tolist()
        )
        df_cat = df_cat.add_prefix(col + "_")
        result = pd.concat([df, df_cat], axis=1)
        return result


class OmvTransformer(BaseEstimator, TransformerMixin):
    """
    Remove records that do not have OMV - other features such as arf, parf, depreciation, etc,
    depend on OMV and so those records with no OMV add no value to our training data
    
    We drop the records as there is only 41 such rows. Alternatively, we can try to see if there
    are other similar vehicles (with make, model, manufacturing year) and compute the OMV from
    those records
    """

    def fit(self, X):
        return self

    def transform(self, X):
        transformed_x = X.copy()
        null_omvs = transformed_x["omv"].isnull()

        logging.info(f"Found {len(null_omvs)} rows with null OMV values")

        transformed_x = transformed_x[~null_omvs]
        return transformed_x


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
        combined_x.coe.replace(10.0, self.mean_coe_per_year.loc[2021].coe, inplace=True)

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
        # Compute arf from omv for those records that have null arf
        rows_without_arf = X[X["arf"].isnull()]
        self.computed_arf = rows_without_arf["omv"].apply(self.compute_arf)
        self.computed_arf.rename("arf_computed", inplace=True)
        return self

    def transform(self, X):
        modified_x = X.join(self.computed_arf)
        modified_x["arf"].fillna(modified_x["arf_computed"], inplace=True)
        modified_x.drop("arf_computed", axis=1, inplace=True)

        return modified_x


class AgeFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Adds a new column "vehicle_age" as min("manufactured", "reg_date_year")
    """

    def fit(self, X):
        self.vehicle_age = datetime.now().year - np.min(
            X[["manufactured", "reg_date_year"]], axis=1
        )
        self.vehicle_age = self.vehicle_age
        return self

    def transform(self, X):
        modified_x = X.copy()
        modified_x["vehicle_age"] = self.vehicle_age
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
        self.parf = X.copy()
        # If the car category does not contain "parf car" then it does not get any part rebate
        self.parf["is_parf_car"] = self.parf.category.apply(
            lambda value: 1 if "parf car" in value else 0
        )
        self.parf = self.parf.apply(self.compute_parf, axis=1)
        return self

    def transform(self, X):
        modified_x = X.copy()
        modified_x["parf"] = self.parf
        return modified_x


class CoeStartDateFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Adds a new column coe_start_date based on reg_date and coe
    For a few
    """

    def fit(self, X):
        self.coe_df = X[["reg_date", "original_reg_date", "coe", "dereg_value"]].copy()

        # Consider original_reg_date/reg_date as coe_start_date in general
        self.coe_df["coe_start_date"] = np.where(
            X["reg_date"].isnull(), X["original_reg_date"], X["reg_date"]
        )
        self.coe_df["coe_start_date"] = pd.to_datetime(self.coe_df["coe_start_date"])
        # Some rows have coe values as 10 - https://www.sgcarmart.com/used_cars/info.php?ID=1027957 (scraping error)
        # In such cases, consider DATASET_GENERATION_DATE as coe_start_date
        self.coe_df.loc[
            self.coe_df["coe"] == 10, "coe_start_date"
        ] = DATASET_GENERATION_DATE

        # Compute coe_expiry date and months left
        self.coe_df["coe_expiry"] = self.coe_df.coe_start_date + np.timedelta64(10, "Y")
        self.coe_df["coe_expiry_months"] = (
            self.coe_df.coe_expiry - DATASET_GENERATION_DATE
        ) / np.timedelta64(1, "M")

        # If the coe expiry is in the past (incorrect), set it as 0
        self.coe_df.coe_expiry_months.clip(lower=0, inplace=True)

        # For rows with incorrect coe_start_date, compute it from dereg_value
        # These rows are not eligible for parf so it can be assumed that dereg_value == coe_rebate for such rows
        # cleaned_df[cleaned_df.coe_expiry_months == 0][cleaned_df.dereg_value == cleaned_df.coe_rebate]
        filter_mask = (self.coe_df.coe_expiry_months == 0) & (
            ~self.coe_df.dereg_value.isnull()
        )
        filtered_df = self.coe_df[filter_mask].copy()

        filtered_df["coe_expiry_months_computed"] = (
            filtered_df.dereg_value * 120
        ) / filtered_df.coe
        filtered_df[
            "coe_expiry_standardized"
        ] = filtered_df.coe_expiry_months_computed.apply(
            lambda value: np.timedelta64(int(value), "M")
        )
        filtered_df["coe_start_date_computed"] = (
            filtered_df["coe_expiry_standardized"] + DATASET_GENERATION_DATE
        ) - np.timedelta64(10, "Y")

        self.coe_df.loc[filter_mask, "coe_start_date"] = filtered_df[
            "coe_start_date_computed"
        ]
        self.coe_df.loc[filter_mask, "coe_expiry_months"] = filtered_df[
            "coe_expiry_months_computed"
        ]

        return self

    def transform(self, X):
        modified_x = X.copy()
        modified_x["coe_start_date"] = self.coe_df["coe_start_date"]
        modified_x["coe_start_year"] = self.coe_df["coe_start_date"].dt.year
        modified_x["coe_expiry_months"] = self.coe_df["coe_expiry_months"]
        return modified_x


class CoeRebateFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Adds a new column "coe_rebate" based on "parf", "coe", and "reg_date"
    """

    def fit(self, X):
        self.x = X.copy()

        self.x["coe_rebate"] = (self.x.coe * self.x.coe_expiry_months) / 120

        # If the computed coe_rebate is 0 (those records for which the coe_start_date is incorrect),
        # use dereg_value as coe_rebate.
        # r.loc[(r.coe_rebate == 0) & (r.dereg_value != 0)]
        # All of these records are for cars that are older than 10 years (no ARF)
        # and so this should be completely safe
        #
        # NOTE: One exception is https://www.sgcarmart.com/used_cars/info.php?ID=1029135 where the
        # coe > coe_rebate - this needs further investigation
        self.x["coe_rebate"] = np.where(
            (self.x["coe_rebate"] == 0) & (self.x["dereg_value"] != 0),
            self.x["dereg_value"],
            self.x["coe_rebate"],
        )

        return self

    def transform(self, X):
        modified_x = X.copy()
        modified_x["coe_rebate"] = self.x["coe_rebate"]

        return modified_x


class DeregValueComputedFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Adds a column dereg_value_computed based on coe_rebate and parf
    """

    def fit(self, X):
        self.dereg_value = X["coe_rebate"] + X["parf"]
        return self

    def transform(self, X):
        modified_x = X.copy()
        modified_x["dereg_value_computed"] = self.dereg_value
        return modified_x


class DeregValueTransformer(BaseEstimator, TransformerMixin):
    """
    Imputes missing dereg_value values based on its corresponding dereg_value_computed
    """

    def fit(self, X):
        return self

    def transform(self, X):
        modified_x = X.copy()
        modified_x["dereg_value"] = np.where(
            X["dereg_value"].isnull(), X["dereg_value_computed"], X["dereg_value"],
        )

        # Drop rows for which it is not possible to compute dereg_value
        # This amounts to 228 rows. Example: https://www.sgcarmart.com/used_cars/info.php?ID=1031249
        # The above is due to error in scraping script which outside the scope of our problem
        modified_x = modified_x.loc[~modified_x["dereg_value"].isnull()]
        return modified_x


class DepreciationTransformer(BaseEstimator, TransformerMixin):
    """
    Imputes missing depreciation values based on its corresponding price and parf
    """

    def fit(self, X):
        return self

    def transform(self, X):
        modified_x = X.copy()
        depreciation_mask = X.depreciation.isnull()
        # Ideally, this should be (price - parf) / no_of_coe_years_left but this formula gives
        # depreciation which are vastly different to the ones in the given dataset - because of incorrect coe_start_date
        # which in turn is due to a scraping error in dataset generation
        modified_x.loc[depreciation_mask, "depreciation"] = (
            X.loc[depreciation_mask, "price"] - X.loc[depreciation_mask, "parf"]
        ) / 10
        return modified_x


class HierarchicalGroupImputer(BaseEstimator, TransformerMixin):
    """
    For missing values in the given feature, this imputer tries filling such values with the agg value
    derived from each group and as fallback uses the entire feature columns' agg value
    """

    RSUFFIX = "_computed"

    def __init__(self, feature, groups, agg_type):
        super(HierarchicalGroupImputer, self).__init__()
        self.feature = feature
        self.groups = groups
        # Order groups from most-specific to least-specific
        self.groups.sort(key=len, reverse=True)
        self.agg_type = agg_type

    def fit(self, X):
        self.modified_x = X.copy()
        for group in self.groups:
            agg_result = self.modified_x.groupby(group).agg(
                {self.feature: self.agg_type}
            )
            feature_computed = self.modified_x.join(
                agg_result, on=group, rsuffix=self.RSUFFIX
            )[self.feature + self.RSUFFIX]
            self.modified_x[self.feature] = np.where(
                self.modified_x[self.feature].isnull(),
                feature_computed,
                self.modified_x[self.feature],
            )

            if not len(self.modified_x[self.modified_x[self.feature].isnull()]):
                break

        # If there are still empty values after the above, fill those values with the
        # column-level agg value
        if len(self.modified_x[self.modified_x[self.feature].isnull()][self.feature]):
            self.modified_x.loc[
                self.modified_x[self.feature].isnull(), self.feature
            ] = self.modified_x[self.feature].agg(self.agg_type)

        return self

    def transform(self, X):
        modified_x = X.copy()
        modified_x[self.feature] = self.modified_x[self.feature]
        return modified_x


class OpcSchemeTransformer(BaseEstimator, TransformerMixin):
    """
    Standardizes and imputes opc_scheme values
    """

    REVISED_OPC = "revised_opc"
    NORMAL_OPC = "normal_opc"
    OLD_OPC = "old_opc"

    def fit(self, X):
        self.opc_scheme = X["opc_scheme"].copy()

        self.opc_scheme.replace(
            "revised opc scheme . learn more about opc schemes.",
            self.REVISED_OPC,
            inplace=True,
        )

        # Only record with this value - https://www.sgcarmart.com/used_cars/info.php?ID=989043
        self.opc_scheme.replace("1100", self.REVISED_OPC, inplace=True)

        self.opc_scheme.fillna(self.NORMAL_OPC, inplace=True)

        self.opc_scheme.replace(
            "old opc scheme . learn more about opc schemes.", self.OLD_OPC, inplace=True
        )

        return self

    def transform(self, X):
        modified_x = X.copy()
        modified_x["opc_scheme"] = self.opc_scheme
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
        modified_x.loc[modified_x["lifespan"].isnull(), "lifespan_restriction"] = 1

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
        self.new_feature = pd.Series(np.zeros(len(X)), index=X.index, dtype=np.int16)
        self.new_feature.loc[~X[self.feature].isnull()] = X[~X[self.feature].isnull()][
            self.feature
        ].apply(lambda value: len(value.split(self.separator)))
        return self

    def transform(self, X):
        modified_x = X.copy()
        modified_x[self.new_feature_name] = self.new_feature
        return modified_x

class CarSpecificationsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, col, group_cols, agg='mean'):
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
            if self.agg == 'mean':
                group_mapping = df[~df[col].isnull()].groupby(group_cols).mean()[col].to_dict()
            elif self.agg == 'mode':
                group_mapping = df[~df[col].isnull()].groupby(group_cols).agg(lambda x: scipy.stats.mode(x)[0])[col].to_dict()
            self.group_mapping_list.append(group_mapping)
            group_cols.pop()
        return self

    def transform(self, input_df):
        group_cols = self.group_cols.copy()
        df = input_df.copy()
        for group_mapping in self.group_mapping_list:
            col = self.col
            
            if col is not None and col in df.columns:
                result = df.apply(lambda row: group_mapping.get(self.get_key(row, group_cols)) if pd.isnull(row[col]) else row[col],
                axis=1)
                df.loc[:, col] = result
            group_cols.pop()
        return df
    
class CarSpecsMissingWithTypeOfVehicle(BaseEstimator, TransformerMixin):
    def __init__(self, cols, agg='mean'):
        self.group_mapping = {}
        self.group_mapping_list = []
        self.cols = cols
        self.agg = agg 
        
    def fit(self, df):
        for col in self.cols:
            if self.agg == 'mean':
                group_mapping = df[~df[col].isnull()].groupby('type_of_vehicle').mean()[col].to_dict()
            elif self.agg == 'mode':
                group_mapping = df[~df[col].isnull()].groupby('type_of_vehicle').agg(lambda x: scipy.stats.mode(x)[0])[col].to_dict()
            self.group_mapping_list.append(group_mapping)
        return self
        
    def transform(self, input_df):
        cols = self.cols
        df = input_df.copy()
        group_mapping_list = self.group_mapping_list
        
        for i in range(len(group_mapping_list)):
            col = cols[i]
            if col is not None and col in df.columns:
                result = df.apply(lambda row: group_mapping_list[i].get(row['type_of_vehicle']) if pd.isnull(row[col]) else row[col],
                axis=1)
                df.loc[:, col] = result
        return df