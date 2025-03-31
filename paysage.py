import os
import re
import pandas as pd
import numpy as np
import time
import webbrowser
import pdb
import copy

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from IPython.display import display
from functools import reduce
from scipy.stats import ks_2samp

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('colheader_justify', 'center')

def dq_report(data, target=None, html=False, csv_engine="pandas", verbose=0):

    correlation_threshold = 0.8
    leakage_threshold = 0.8

    if not verbose:
        print("This is a summary report. Change verbose to 1 to see more details on each DQ issue.")

    if isinstance(target, str):
        if target == '':
            target = None        

    if isinstance(data, str):
        ext = os.path.splitext(data)[-1]

        if ext == ".csv":
            print("    If large dataset, we will randomly sample 100k rows to speed up processing...")
            if csv_engine == 'pandas':
                df = pd.read_csv(data)
            elif csv_engine == 'polars':
                import polars as pl
                df = pl.read_csv(data)
            elif csv_engine == 'parquet':
                import pyarrow as pa
                df = pa.read_table(data)
            else :
                if str(pd.__version__)[0] == '2':
                    print(f"    pandas version={pd.__version__}. Using pyarrow backend.")
                    df = pd.read_csv(data, engine='pyarrow', dtype_backend='pyarrow')
                else:
                    print(f"    pandas version={pd.__version__}. Using pandas backend.")
                    df = pd.read_csv(data)
        elif ext == ".parquet":
            df = pd.read_parquet(data)
        elif ext in [".feather", ".arrow", "ftr"]:
            df = pd.read_feather(data)
        else:
            print("    Unsupported file format. Please use CSV, parquet, feather or arrow.")
            return data

        if df.shape[0] >= 1000000:
            df = df.sample(100000)

    elif isinstance(data, pd.DataFrame):
        df = copy.deepcopy(data)
    else:
        print("    Unrecognized input. Please provide a filename or a pandas dataframe...")
        return data

    dup_rows = df.duplicated().sum()
    if dup_rows > 0:
        print(f'There are {dup_rows} duplicate rows in your dataset')
        print(f'    Alert: Dropping duplicate rows can sometimes cause your column data types to change to object.')
        df = df.drop_duplicates()
    
    dup_cols = df.columns[df.columns.duplicated()]
    if len(dup_cols) > 0:
        print(f'    Alert: Dropping {len(dup_cols)} duplicate cols')
        df = df.T[df.T.index.duplicated(keep='first')].T

    new_col = 'DQ Issue'
    good_col = "The Good News"
    bad_col = "The Bad News"

    dq_df1 = pd.DataFrame(columns=[good_col, bad_col])
    dq_df1 = dq_df1.T
    dq_df1["first_comma"] = ""
    dq_df1[new_col] = ""

    data_types = pd.DataFrame(
        df.dtypes,
        columns=['Data Type']
    )

    missing_values = df.isnull().sum()
    missing_values_pct = ((df.isnull().sum()/df.shape[0])*100)
    missing_cols = missing_values[missing_values > 0].index.tolist()

    if not target is None:
        var_df = classify_columns(df.drop(target, axis=1), verbose=0)
    else:
        var_df = classify_columns(df, verbose=0) 
    
    IDcols = var_df['id_vars']
    nlp_vars = var_df['nlp_vars']
    discrete_string_vars = var_df['discrete_string_vars']
    cols_delete = var_df['cols_delete']
    bool_vars = var_df['string_bool_vars'] + var_df['num_bool_vars']
    int_vars = var_df['int_vars']
    categorical_vars = var_df['cat_vars'] + var_df['factor_vars'] 
    date_vars = var_df['date_vars']

    if target is None:
        preds = [x for x in list(df) if x not in IDcols+cols_delete]
    else:
        if isinstance(target, str):
            preds = [x for x in list(df) if x not in IDcols+cols_delete+[target]]
        else:
            preds = [x for x in list(df) if x not in IDcols+cols_delete+target]
     
    float_cols = var_df['continuous_vars']
    id_cols = list(set(IDcols[:]))
    zero_var_cols = list(set(cols_delete[:]))
    number_cols = list(set(var_df['continuous_vars'] + var_df['int_vars']))
    text_vars = list(set(discrete_string_vars + nlp_vars))
    cat_cols = categorical_vars[:]
    date_cols = date_vars[:]

    missing_data = pd.DataFrame(
        missing_values_pct,
        columns=['Missing Values%']
    )
    unique_values = pd.DataFrame(
        columns=['Unique Values%']
    )

    for col in list(df.columns.values):
        if col in float_cols:
            unique_values.loc[col] = ["NA"]
        else:
            unique_values.loc[col] = [int(100*df[col].nunique()/df.shape[0])]
      
    maximum_values = pd.DataFrame(
        columns=['Maximum Value']
    )
    minimum_values = pd.DataFrame(
        columns=['Minimum Value']
    )

    for col in list(df.columns.values):
        if col not in missing_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                maximum_values.loc[col] = [df[col].max()]
        elif col in number_cols:
            maximum_values.loc[col] = [df[col].max()]

    for col in list(df.columns.values):
        if col not in missing_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                minimum_values.loc[col] = [df[col].min()]
        elif col in number_cols:
            minimum_values.loc[col] = [df[col].min()]
    
    dq_df2 = data_types.join(missing_data).join(unique_values).join(minimum_values).join(maximum_values)
    dq_df2['Minimum Value'] = dq_df2[['Minimum Value']].fillna('')
    dq_df2['Maximum Value'] = dq_df2[['Maximum Value']].fillna('')
 
    dq_df2["first_comma"] = ""
    dq_df2[new_col] = f""
    
    if dup_rows > 0:
        new_string =  f"There are {dup_rows} duplicate rows in the dataset. De-dup these rows using Fix_DQ."
        dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
        dq_df1.loc[bad_col,'first_comma'] = ', '
    else:
        new_string =  f"There are no duplicate rows in this dataset"
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '

    if len(dup_cols) > 0:
        new_string =  f"There are {len(dup_cols)} duplicate columns in the dataset. De-dup {dup_cols} using Fix_DQ."
        dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
        dq_df1.loc[bad_col,'first_comma'] = ', '
    else:
        new_string =  f"There are no duplicate columns in this datatset"
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '

    if len(id_cols) > 0:
        new_string = f"There are ID columns in the dataset. Remove them before modeling using Fix_DQ."
        dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
        dq_df1.loc[bad_col,'first_comma'] = ', '

        for col in id_cols:
            new_string = f"Possible ID column: drop before modeling step."
            dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
            dq_df2.loc[col,'first_comma'] = ', '
    else:
        new_string = f"There are no ID columns in the dataset."
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '

    if len(zero_var_cols) > 0:
        new_string = f"These are zero-variance or low information columns in the dataset. Remove them before modeling."
        dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
        dq_df1.loc[bad_col,'first_comma'] = ', '

        for col in zero_var_cols:
            new_string = f"Possible Zero-variance or low information colum: drop before modeling step."
            dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
            dq_df2.loc[col,'first_comma'] = ', '
    else:
        new_string = f"There are no zero-variance or low information columns in the dataset."
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '

    if len(date_cols) > 0:
        new_string =  f"There are {len(date_vars)} date-time vars in the dataset. Make sure you transform them before modeling."
        dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
        dq_df1.loc[bad_col,'first_comma'] = ', '

        for col in date_cols:
            new_string = f"Possible date-time colum: transform before modeling step."
            dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
            dq_df2.loc[col,'first_comma'] = ', '
    else:
        new_string =  f"There are no date-time vars in this dataset"
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '

    if len(missing_cols) > 0:
        for col in missing_cols:
            if missing_values[col] > 0:
                new_string = f"{missing_values[col]} missing values. Impute them with mean, median, mode, or a constant value such as 123."
                dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
                dq_df2.loc[col,'first_comma'] = ', '
    else:
        new_string = f"There are no columns with missing values in the dataset"
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '
    
    rare_threshold = 0.01
    rare_cat_cols = []

    if len(cat_cols) > 0:
        for col in cat_cols:
            value_counts = df[col].value_counts(normalize=True)
            rare_values = value_counts[value_counts < rare_threshold].index.tolist()

            if len(rare_values) > 0:
                rare_cat_cols.append(col)

                if len(rare_values) <= 10:
                    new_string = f"{len(rare_values)} rare categories: {rare_values}. Group them into a single category or drop the categories."
                else:
                    new_string = f"{len(rare_values)} rare categories: Too many to list. Group them into a single category or drop the categories."

                dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
                dq_df2.loc[col,'first_comma'] = ', '
    else:
        new_string =  f"There are no categorical columns with rare categories (< {100*rare_threshold:.0f} percent) in this dataset"
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '

    inf_values = df.replace([np.inf, -np.inf], np.nan).isnull().sum() - missing_values
    inf_cols = inf_values[inf_values > 0].index.tolist()

    if len(inf_cols) > 0:
        new_string =  f"There are {len(inf_cols)} columns with infinite values in the dataset. Replace them with NaN or a finite value."
        dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
        dq_df1.loc[bad_col,'first_comma'] = ', '

        for col in inf_cols:
            if inf_values[col] > 0:
                new_string = f"{inf_values[col]} infinite values. Replace them with a finite value."
                dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
                dq_df2.loc[col,'first_comma'] = ', '
    else:
        new_string =  f"There are no columns with infinite values in this dataset "
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '

    mixed_types = df[preds].applymap(type).nunique()
    mixed_cols = mixed_types[mixed_types > 1].index.tolist()

    if len(mixed_cols) > 0:
        new_string = f"There are {len(mixed_cols)} columns with mixed data types in the dataset. Convert them to a single type or split them into multiple columns."
        dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
        dq_df1.loc[bad_col,'first_comma'] = ', '

        for col in mixed_cols:
            if mixed_types[col] > 1:
                new_string = f"Mixed dtypes: has {mixed_types[col]} different data types: "
                for each_class in df[col].apply(type).unique():
                    if each_class == str:
                        new_string +=  f" object,"
                    elif each_class == int:
                        new_string +=  f" integer,"
                    elif each_class == float:
                        new_string +=  f" float,"
                dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
                dq_df2.loc[col,'first_comma'] = ', '
    else:
        new_string =  f"There are no columns with mixed (more than one) dataypes in this dataset"
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '
       
    num_cols = var_df['continuous_vars'] + var_df['int_vars']

    if len(num_cols) > 0:
        first_time = True
        outlier_cols = []

        for col in num_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]

            if len(outliers) > 0:
                outlier_cols.append(col)

                if first_time:
                    new_string = f"There are {len(num_cols)} numerical columns, some with outliers. Remove them or use robust statistics."
                    dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
                    dq_df1.loc[bad_col,'first_comma'] = ', '
                    first_time =False

                new_string = f"Column has {len(outliers)} outliers greater than upper bound ({upper_bound:.2f}) or lower than lower bound({lower_bound:.2f}). Cap them or remove them."
                dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
                dq_df2.loc[col,'first_comma'] = ', '

        if len(outlier_cols) < 1:
            new_string =  f"There are no numeric columns with outliers in this dataset"
            dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
            dq_df1.loc[good_col,'first_comma'] = ', '
                
    cardinality = df[discrete_string_vars].nunique()
    cardinality_threshold = min(30, cardinality.min())
    high_card_cols = discrete_string_vars[:] 

    if len(high_card_cols) > 0:
        new_string = f"There are {len(high_card_cols)} columns with high cardinality (>{cardinality_threshold} categories) in the dataset. Reduce them using encoding techniques or feature selection methods."
        dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
        dq_df1.loc[bad_col,'first_comma'] = ', '
        for col in high_card_cols:
            new_string = f"Possible high cardinality column with {cardinality[col]} unique values: Use hash encoding or text embedding to reduce dimension."
            dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
            dq_df2.loc[col,'first_comma'] = ', '
    else:
        new_string =  f"There are no high cardinality columns in this dataset"
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '

    correlation_matrix = df[num_cols].corr().abs()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    high_corr_cols = [column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)]

    if len(high_corr_cols) > 0:
        new_string = f"There are {len(high_corr_cols)} columns with >= {correlation_threshold} correlation in the dataset. Drop one of them or use dimensionality reduction techniques."
        dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
        dq_df1.loc[bad_col,'first_comma'] = ', '

        for col in high_corr_cols:
            new_string = f"Column has a high correlation with {upper_triangle[col][upper_triangle[col] > correlation_threshold].index.tolist()}. Consider dropping one of them."
            dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
            dq_df2.loc[col,'first_comma'] = ', '
    else:
        new_string =  f"There are no highly correlated columns in the dataset."
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '

    if target is not None:
        if isinstance(target, str):
            target_col = [target]
        else:
            target_col = copy.deepcopy(target)
        
        cat_cols = df[target_col].select_dtypes(include=["object", "category"]).columns.tolist() 
        
        model_type = 'Regression'

        if len(cat_cols) > 0:
            model_type =  "Classification"
        else:
            int_cols = df[target_col].select_dtypes(include=["integer"]).columns.tolist() 
            copy_target_col = copy.deepcopy(target_col)
            for each_target_col in copy_target_col:
                if len(df[each_target_col].value_counts()) <= 30:
                    model_type =  "Classification"
        
        if model_type == 'Classification':
            for each_target_col in target_col:
                y = df[each_target_col]
                value_counts = y.value_counts(normalize=True)
                min_freq = value_counts.min()
                max_freq = value_counts.max()
                imbalance_threshold = 0.1

                if min_freq < imbalance_threshold or max_freq > 1 - imbalance_threshold:
                    new_string =  f"Imbalanced classes in target variable ({each_target_col}). Use resampling or class weights to address."
                    dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
                    dq_df1.loc[bad_col,'first_comma'] = ', '
                    dq_df2.loc[each_target_col, new_col] += "Target column. Appears to have Imbalanced classes. Try balancing classes."
            
        leakage_matrix = df[preds].corrwith(df[target_col]).abs()
        leakage_cols = leakage_matrix[leakage_matrix > leakage_threshold].index.tolist()

        if len(leakage_cols) > 0:
            new_string = f"There are {len(leakage_cols)} columns with data leakage. Double check whether you should use this variable."
            dq_df1.loc[bad_col,new_col] += dq_df1.loc[bad_col,'first_comma'] + new_string
            dq_df1.loc[bad_col,'first_comma'] = ', '
            for col in leakage_cols:
                new_string = f"    {col} has a correlation >= {leakage_threshold} with {target_col}. Possible data leakage. Double check this variable."
                dq_df2.loc[col,new_col] += dq_df2.loc[col,'first_comma'] + new_string
                dq_df2.loc[col,'first_comma'] = ', '
        else:
            new_string =  f'There are no target leakage columns in the dataset'
            dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
            dq_df1.loc[good_col,'first_comma'] = ', '
    else:
        new_string = f'There is no target given. Hence no target leakage columns detected in the dataset'
        dq_df1.loc[good_col,new_col] += dq_df1.loc[good_col,'first_comma'] + new_string
        dq_df1.loc[good_col,'first_comma'] = ', '
        target_col = []

    dq_df1.drop('first_comma', axis=1, inplace=True)
    dq_df2.drop('first_comma', axis=1, inplace=True)

    for col in list(df):
        if dq_df2.loc[col, new_col] == "":
            if col in target_col:
                if df[col].nunique() == 1:
                    dq_df2.loc[col,new_col] += "Target column. Appears to have zero variance. Double-check it."
                else:
                    dq_df2.loc[col,new_col] += "Target column"
            else:
                dq_df2.loc[col,new_col] += "No issue"

    if html:
        if verbose == 0:
            write_to_html(dq_df1, "dq_report.html")
        else:
            write_to_html(dq_df2, "dq_report.html")
    else:
        try:
            from IPython.display import display
        except Exception as e:
            print('Error due to %s. Please install and try again...')
            return dq_df2
        if verbose < 0:
            pass
        elif verbose == 0:
            all_rows = dq_df1.shape[0]
            ax = dq_df1.head(all_rows).style.background_gradient(cmap='Reds').set_properties(**{'font-family': 'Segoe UI'})
            display(ax);
        else:
            all_rows = dq_df2.shape[0]
            ax = dq_df2.head(all_rows).style.background_gradient(cmap='Reds').set_properties(**{'font-family': 'Segoe UI'})
            display(ax);

    return dq_df2

def write_to_html(dqr, filename="dq_report.html"):

    df_html = dqr.to_html(classes="table table-striped table-bordered table-hover",
                border=0, na_rep="", index=True).replace('<th>', 
                '<th style="background-color: lightgreen">').replace('<td>', 
                '<td style="color: blue">')

    df_html = f""" <style> @import url(‘https://fonts.googleapis.com/css?family=Roboto&display=swap’);
        table {{ font-family: Roboto; font-size: 12px; }}
        th {{ background-color: orange; font-size: 14px; text-align: center; }}
        td {{ color: blue; font-style: italic; text-align: left; }}
        tr:nth-child(odd) {{ background-color: lightyellow; }}
        tr:nth-child(even) {{ background-color: lightgrey; }} </style> {df_html} 
        """

    with open(filename, "w") as f:
        f.write(df_html)

    webbrowser.open_new_tab(filename)

def left_subtract(l1,l2):
    lst = []
    for i in l1:
        if i not in l2:
            lst.append(i)
    return lst

def compare_unique(df1, df2, column):
    set1 = set(df1[column].unique())
    set2 = set(df2[column].unique())

    count_1 = len(set1)
    count_2 = len(set2)
    diff_1_2 = list(set1 - set2)
    diff_2_1 = list(set2 - set1)
    
    result = {
        "unique_count_in_df1": count_1,
        "unique_count_in_df2": count_2,
        "diff_between_df1_df2": diff_1_2,
        "diff_between_df2_df1": diff_2_1,
    }

    return result

class Fix_DQ(BaseEstimator, TransformerMixin):
    def __init__(self, quantile=0.87, cat_fill_value="missing", num_fill_value=9999, 
                 rare_threshold=0.01, correlation_threshold=0.9):
        self.quantile = quantile
        self.cat_fill_value = cat_fill_value
        self.num_fill_value = num_fill_value
        self.rare_threshold = rare_threshold
        self.correlation_threshold = correlation_threshold

    def cap_outliers(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        X = copy.deepcopy(X)
        num_cols = X.select_dtypes(include=[ "float"]).columns.tolist()
        
        for col in num_cols:
            if col in self.upper_bounds_:
                X[col] = np.where(X[col] > self.upper_bounds_[col], self.upper_bounds_[col], X[col])
            else:
                continue

        return X
    
    def impute_missing(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        X = copy.deepcopy(X)
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X.select_dtypes(include=["int", "float"]).columns.tolist()
        
        missing_values = X.isnull().sum()
        missing_cols = missing_values[missing_values > 0].index.tolist()

        for col in missing_cols:
            if not col in self.missing_cols_:
                self.missing_cols_.append(col)

        for col in self.missing_cols_:
            if col in cat_cols:
                if isinstance(self.cat_fill_value, dict):
                    if col in self.cat_fill_value:
                        X[col] = X[[col]].fillna(self.cat_fill_value[col]).values
                    else:
                        X[col] = X[[col]].fillna("missing").values
                else:
                    X[col] = X[[col]].fillna(self.cat_fill_value).values
        
        for col in self.missing_cols_:
            if col in num_cols:
                if isinstance(self.num_fill_value, dict):
                    if col in self.num_fill_value:
                        X[col] = X[[col]].fillna(self.num_fill_value[col]).values
                    else:
                        X[col] = X[[col]].fillna(-999).values
                else:
                    X[col] = X[[col]].fillna(self.num_fill_value).values

        return X

    def group_rare_categories(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
                
        for col in cat_cols:
            value_counts = X[col].value_counts(normalize=True)
            rare_values = value_counts[value_counts < self.rare_threshold].index.tolist()

            if len(rare_values) > 0:
                X[col] = X[col].replace(rare_values, "Rare")
        
        return X
    
    def replace_infinite(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        num_cols = X.select_dtypes(include=["int", "float"]).columns.tolist()
        
        for col in num_cols:
            if col in self.upper_bounds_:
                X[col] = X[col].replace([np.inf, -np.inf], self.upper_bounds_[col])

        return X

    def detect_duplicates(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        dup_rows = X.duplicated().sum()
        if dup_rows > 0:
            print(f'Alert: Detecting {dup_rows} duplicate rows...')
        
        dup_cols = X.columns[X.columns.duplicated()]

        if len(dup_cols) > 0:
            print(f'Alert: Detecting {len(dup_cols)} duplicate cols...')
            X = X.T[X.T.index.duplicated(keep='first')].T
        
        return X

    def drop_duplicated(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        dup_rows = X.duplicated().sum()
        if dup_rows > 0:
            print(f'Alert: Dropping {dup_rows} duplicate rows can sometimes cause column data types to change to object. Double-check!')
            X = X.drop_duplicates(keep='first')
        
        dup_cols = X.columns[X.columns.duplicated()]

        if len(dup_cols) > 0:
            print(f'Alert: Dropping {len(dup_cols)} duplicate cols: {dup_cols}!')
            X = X.T[X.T.index.duplicated(keep='first')].T
        
        return X
    
    def transform_skewed(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        num_cols = X.select_dtypes(include=["float"]).columns.tolist()
                
        for col in num_cols:
            if col in self.col_transformers_:
                if str(self.col_transformers_[col]).split("(")[0] == "PowerTransformer":
                    pt = self.col_transformers_[col]
                    X[col] = pt.transform(X[[col]])
                else:
                    ft = self.col_transformers_[col]
                    X[col] = ft.transform(X[col])

        return X
    
    def fit(self, X, y=None):
        self.drop_cols_ = []
        self.missing_cols_ = []
      
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        num_cols = X.select_dtypes(include=["int", "float"]).columns.tolist()
        float_cols = X.select_dtypes(include=["float"]).columns.tolist()
        non_float_cols = left_subtract(X.columns, float_cols)
        missing_values = X.isnull().sum()
        self.missing_cols_ = missing_values[missing_values > 0].index.tolist()
        drop_missing = []

        for each in self.missing_cols_:
            if X[each].isna().sum()/len(X) >= 0.80 :
                drop_missing.append(each)
                print(f"    Dropping {each} since it has >= 80%% missing values")

        X = self.detect_duplicates(X)

        self.id_cols_ = [column for column in non_float_cols if X[column].nunique() == X.shape[0]]
        if len(self.id_cols_) > 0:
            print(f"    Dropping {len(self.id_cols_)} ID column(s): {self.id_cols_}")

        self.zero_var_cols_ = [column for column in non_float_cols if X[column].nunique() == 1]
        if len(self.zero_var_cols_) > 0:
            print(f"    Dropping {len(self.zero_var_cols_)} zero-variance cols: {self.zero_var_cols_}")
        
        self.drop_corr_cols_ = []
        correlation_matrix = X[num_cols].corr().abs()
        upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        high_corr_cols = [column for column in upper_triangle.columns if any(upper_triangle[column] > self.correlation_threshold)]

        if len(high_corr_cols) > 0:
            self.drop_corr_cols_ = high_corr_cols
            for col in high_corr_cols:
                    print(f"    Dropping {col} which has a high correlation with {upper_triangle[col][upper_triangle[col] > self.correlation_threshold].index.tolist()}")
        
        self.upper_bounds_ = {}
        
        if self.quantile is None:
            base_quantile = 0.99
            for col in float_cols:
                q3 = X[col].quantile(base_quantile)
                iqr = X[col].quantile(base_quantile) - X[col].quantile(1 - base_quantile)
                upper_bound = q3 + 1.5 * iqr
                self.upper_bounds_[col] = upper_bound
        else:
            for col in float_cols:
                q3 = X[col].quantile(self.quantile)
                iqr = X[col].quantile(self.quantile) - X[col].quantile(1 - self.quantile)
                upper_bound = q3 + 1.5 * iqr
                self.upper_bounds_[col] = upper_bound

        self.col_transformers_ = {}
        
        skew_threshold = 1.0
        
        for col in float_cols:
            skewness = X[col].skew()

            if abs(skewness) > skew_threshold:
                if X[col].min() > 0:
                    ft = FunctionTransformer(np.log1p)
                    ft.fit(X[col])
                    self.col_transformers_[col] = ft
                elif X[col].min() > 0 and "scipy" in sys.modules:
                    pt = PowerTransformer(method="box-cox")
                    pt.fit(X[[col]])
                    self.col_transformers_[col] = pt
                else:
                    pt = PowerTransformer(method="yeo-johnson")
                    pt.fit(X[[col]])
                    self.col_transformers_[col] = pt

        self.mixed_type_cols_ = []
        mixed_types = X.applymap(type).nunique()
        self.mixed_type_cols_ = mixed_types[mixed_types > 1].index.tolist()

        if len(self.mixed_type_cols_) > 0:
            extra_mixed = left_subtract(self.mixed_type_cols_, self.missing_cols_)

            if len(extra_mixed) > 0:
                print(f"    Dropping {len(extra_mixed)} columns due to mixed data types")
                for each in extra_mixed:
                    print(f"        {each} has mixed dtypes: {X[each].apply(type).unique()}")    

        if len(self.id_cols_) > 0:
            self.drop_cols_ += self.id_cols_

        if len(self.zero_var_cols_) > 0:
            self.drop_cols_ += self.zero_var_cols_

        if len(self.mixed_type_cols_) > 0:
            drop_cols = left_subtract(extra_mixed, self.zero_var_cols_+self.id_cols_)
            if len(drop_cols) > 0:
                self.drop_cols_ += drop_cols
            if len(extra_mixed) > 0:
                self.drop_cols_ += extra_mixed
            
        if len(self.drop_corr_cols_) > 0:
            if len(left_subtract(self.drop_corr_cols_, self.drop_cols_)) > 0:
                extra_cols = left_subtract(self.drop_corr_cols_,self.drop_cols_)
                self.drop_cols_ += extra_cols

        if len(drop_missing) > 0:
            self.drop_cols_ += drop_missing

        self.drop_cols_ = list(set(self.drop_cols_))

        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = self.drop_duplicated(X)

        if len(self.drop_cols_) > 0:
            X = X.drop(self.drop_cols_, axis=1)
            print(f'Dropped {len(self.drop_cols_)} columns total in dataset')

        imputed_X = self.impute_missing(X)

        if self.quantile is None:
            capped_X = copy.deepcopy(imputed_X)
        else:
            capped_X = self.cap_outliers(imputed_X)

        infinite_X = self.replace_infinite(capped_X)
        rare_X = self.group_rare_categories(infinite_X)
        transformed_X = self.transform_skewed(rare_X)

        return transformed_X

class DataSchemaChecker(BaseEstimator, TransformerMixin):

    def __init__(self, schema):
        self.schema = schema

    def fit(self, df):  
        if len(df.columns) != len(self.schema):
            raise ValueError("The number of columns in the dataframe does not match the number of columns in the schema")

        self.translated_schema = {}

        for column, dtype in self.schema.items():
            if dtype in ["string","object","category", "str"]:
                self.translated_schema[column] = "object"
            elif dtype in ["text","NLP","nlp"]:
                self.translated_schema[column] = "object"
            elif dtype in ["boolean","bool"]:
                self.translated_schema[column] = "bool"
            elif dtype in [ "np.int8", "int8"]:
                self.translated_schema[column] = "int8"
            elif dtype in ["np.int16","int16"]:
                self.translated_schema[column] = "int16"
            elif dtype in ["int32", "np.int32"]:
                self.translated_schema[column] = "int32"
            elif dtype in ["integer","int", "int64", "np.int64"]:
                self.translated_schema[column] = "int64"
            elif dtype in ["date"]:
                self.translated_schema[column] = "datetime64[ns]"                
            elif dtype in ["float"]:
                self.translated_schema[column] = "float64"
            elif dtype in ["np.float32", "float32"]: 
                self.translated_schema[column] = "float32"
            elif dtype in ["np.float64", "float64"]:
                self.translated_schema[column] = "float64"
            else:
                raise ValueError("Invalid data type: {}".format(dtype))

        return self
            
    def transform(self, df):       
        df = copy.deepcopy(df)
        errors = []

        for column, dtype in self.translated_schema.items():
            actual_dtype = df[column].dtype
            
            if actual_dtype != dtype:
                errors.append({
                    "column": column,
                    "expected_dtype": dtype,
                    "actual_dtype": actual_dtype,
                    "data_dtype_mismatch": "Column '{}' has data type '{}' but expected '{}'".format(
                        column, actual_dtype, dtype)})

        if errors:
            self.error_df_ = pd.DataFrame(errors)
            display(self.error_df_)
        else:
            self.error_df_ = pd.DataFrame()
            print("**No Data Schema Errors**")

        if len(self.error_df_) > 0:
            for i, row in self.error_df_.iterrows():
                column = row['column']
                try:
                    if row['expected_dtype']=='datetime64[ns]':
                        df[column] = pd.to_datetime(df[column])
                    else:
                        df[column] = df[column].astype(row["expected_dtype"])
                except:
                    print(f"Converting {column} to {self.error_df_['expected_dtype'][0]} is erroring. Please convert it yourself.")
                
        return df

def dc_report(train, test, exclude=[], html=False, verbose=0):
    train = copy.deepcopy(train)
    test = copy.deepcopy(test)

    if not isinstance(train, pd.DataFrame) or not isinstance(test, pd.DataFrame):
        print("The input must be pandas dataframes. Stopping!")
        return pd.DataFrame()

    if len(exclude) > 0:
        for each in exclude:
            if each in train.columns:
                train = train.drop(each, axis=1)
            else:
                print('Column %s not found in train' %each)
            if each in test.columns:
                test = test.drop(each, axis=1)
            else:
                print('Column %s not found in train' %each)

    if not train.columns.equals(test.columns):
        print("The two dataframes dont have the same columns. Use exclude argument to exclude columns from comparison.")
        return pd.DataFrame()
    else:
        print('Analyzing two dataframes for differences. This will take time, please be patient...')

    dqr_tr = dq_report(data=train, verbose=-1)
    dqr_te = dq_report(data=test,verbose = -1)
    report = dqr_tr.join(dqr_te, lsuffix="_Train", rsuffix="_Test")
    dist_diff = []

    for col in train.columns:
        dtype_train = train[col].dtype
        dtype_test = test[col].dtype
        missing_train = dqr_tr.loc[col, "Missing Values%"]
        missing_test = dqr_te.loc[col, "Missing Values%"]
        unique_train = dqr_tr.loc[col, "Unique Values%"]

        if dqr_tr.loc[col, "Unique Values%"]=='NA':
            count_unique_train = 0
        else:
            count_unique_train = len(train)*(unique_train / 100)

        unique_test = dqr_te.loc[col, "Unique Values%"]

        if dqr_te.loc[col, "Unique Values%"]=='NA':
            count_unique_test = 0
        else:
            count_unique_test = len(test)*(unique_test / 100)

        dist_diff_col = ""
        
        if np.issubdtype(dtype_train, np.number) and np.issubdtype(dtype_test, np.number) and count_unique_train < 10 and count_unique_test < 10:
            min_train = report.loc[ col, "Minimum Value_Train"]
            min_test = report.loc[ col, "Minimum Value_Test"]
            max_train = report.loc[ col, "Maximum Value_Train"]
            max_test = report.loc[ col, "Maximum Value_Test"]

            if missing_train < 100 and missing_test < 100:
                ks_stat = ks_2samp(train[col].dropna(), test[col].dropna()).statistic

                if ks_stat > 0:
                    dist_diff_col += f"The distributions of {col} are different with a KS test statistic of {ks_stat:.3f}. "

        if missing_train != missing_test:
            dist_diff_col += f"The percentage of missing values of {col} are different between train ({missing_train:.2f}%) and test ({missing_test:.2f}%). "

        if unique_train != unique_test:
            if unique_train=='NA' or unique_test == 'NA':
                dist_diff_col += f"The data types of {col} are different between train: {train[col].dtype} and test: {test[col].dtype}. "
            else:
                dist_diff_col += f"The percentage of unique values of {col} are different between train ({unique_train:.2f}%) and test ({unique_test:.2f}%). "
        
        if dist_diff_col == "":
            dist_diff_col = None

        dist_diff.append(dist_diff_col)

    report["Distribution Difference"] = dist_diff
    report = report.reset_index().rename(columns={'index':"Column Name"})

    if verbose:
        if html:
            write_to_html(report, filename="dc_report.html")
        else:  
            all_rows = report.shape[0]
            ax = report.head(all_rows).style.background_gradient(cmap='Reds').set_properties(**{'font-family': 'Segoe UI'})
            display(ax);
        return report
    else:
        short_report = report[['Column Name','DQ Issue_Train','DQ Issue_Test',"Distribution Difference"]]
        if html:
            write_to_html(short_report, filename="dc_report.html")
        else:
            all_rows = short_report.shape[0]
            ax = short_report.head(all_rows).style.background_gradient(cmap='Reds').set_properties(**{'font-family': 'Segoe UI'})
            display(ax);
        return short_report

def classify_columns(df_preds, verbose=0):
    train = copy.deepcopy(df_preds)
    max_nlp_char_size = 30
    max_cols_to_print = 30
    cat_limit = 35
    float_limit = 15 

    def add(a,b):
        return a+b

    sum_all_cols = dict()
    orig_cols_total = train.shape[1]
    
    cols_delete = []
    cols_delete = [col for col in list(train) if (len(train[col].value_counts()) == 1
                                       ) | (train[col].isnull().sum()/len(train) >= 0.90)]
    inf_cols = EDA_find_remove_columns_with_infinity(train, remove=False, verbose=verbose)
    mixed_cols = [x for x in list(train) if len(train[x].dropna().apply(type).value_counts()) > 1]

    if len(mixed_cols) > 0:
        print('    Removing %s column(s) due to mixed data type detected...' %mixed_cols)

    cols_delete += mixed_cols
    cols_delete += inf_cols
    train = train[left_subtract(list(train),cols_delete)]
    var_df = pd.Series(dict(train.dtypes)).reset_index(drop=False).rename(
                        columns={0:'type_of_column'})
    sum_all_cols['cols_delete'] = cols_delete

    var_df['bool'] = var_df.apply(lambda x: 1 if x['type_of_column'] in ['bool','object']
                        and len(train[x['index']].value_counts()) == 2 else 0, axis=1)
    string_bool_vars = list(var_df[(var_df['bool'] ==1)]['index'])
    sum_all_cols['string_bool_vars'] = string_bool_vars
    var_df['num_bool'] = var_df.apply(lambda x: 1 if x['type_of_column'] in [np.uint8,
                            np.uint16, np.uint32, np.uint64,
                            'int8','int16','int32','int64',
                            'float16','float32','float64'] and len(
                        train[x['index']].value_counts()) == 2 else 0, axis=1)
    num_bool_vars = list(var_df[(var_df['num_bool'] ==1)]['index'])
    sum_all_cols['num_bool_vars'] = num_bool_vars
    
    discrete_or_nlp = var_df.apply(lambda x: 1 if x['type_of_column'] in ['object']  and x[
        'index'] not in string_bool_vars+cols_delete else 0,axis=1)
    
    var_df['nlp_strings'] = 0
    var_df['discrete_strings'] = 0
    var_df['cat'] = 0
    var_df['id_col'] = 0
    discrete_or_nlp_vars = var_df.loc[discrete_or_nlp==1]['index'].values.tolist()
    copy_discrete_or_nlp_vars = copy.deepcopy(discrete_or_nlp_vars)

    if len(discrete_or_nlp_vars) > 0:
        for col in copy_discrete_or_nlp_vars:
            train[[col]] = train[[col]].fillna('  ')

            if train[col].map(lambda x: len(x) if type(x)==str else 0).max(
                ) >= 50 and len(train[col].value_counts()
                        ) >= int(0.9*len(train)) and col not in string_bool_vars:
                var_df.loc[var_df['index']==col,'nlp_strings'] = 1
            elif train[col].map(lambda x: len(x) if type(x)==str else 0).mean(
                ) >= max_nlp_char_size and train[col].map(lambda x: len(x) if type(x)==str else 0).max(
                ) < 50 and len(train[col].value_counts()
                        ) <= int(0.9*len(train)) and col not in string_bool_vars:
                var_df.loc[var_df['index']==col,'discrete_strings'] = 1
            elif len(train[col].value_counts()) > cat_limit and len(train[col].value_counts()
                        ) <= int(0.9*len(train)) and col not in string_bool_vars:
                var_df.loc[var_df['index']==col,'discrete_strings'] = 1
            elif len(train[col].value_counts()) > cat_limit and len(train[col].value_counts()
                        ) == len(train) and col not in string_bool_vars:
                var_df.loc[var_df['index']==col,'id_col'] = 1
            else:
                var_df.loc[var_df['index']==col,'cat'] = 1

    nlp_vars = list(var_df[(var_df['nlp_strings'] ==1)]['index'])
    sum_all_cols['nlp_vars'] = nlp_vars
    discrete_string_vars = list(var_df[(var_df['discrete_strings'] ==1) ]['index'])
    sum_all_cols['discrete_string_vars'] = discrete_string_vars
    var_df['dcat'] = var_df.apply(lambda x: 1 if str(x['type_of_column'])=='category' else 0,
                            axis=1)
    factor_vars = list(var_df[(var_df['dcat'] ==1)]['index'])
    sum_all_cols['factor_vars'] = factor_vars
    
    date_or_id = var_df.apply(lambda x: 1 if x['type_of_column'] in [np.uint8,
                         np.uint16, np.uint32, np.uint64,
                         'int8','int16',
                        'int32','int64']  and x[
        'index'] not in string_bool_vars+num_bool_vars+discrete_string_vars+nlp_vars else 0,
                                        axis=1)
    
    var_df['int'] = 0
    var_df['date_time'] = 0
    
    var_df['date_time'] = var_df.apply(lambda x: 1 if x['type_of_column'] in ['<M8[ns]','datetime64[ns]']  and x[
        'index'] not in string_bool_vars+num_bool_vars+discrete_string_vars+nlp_vars else 0,
                                        axis=1)
    
    if len(var_df.loc[date_or_id==1]) != 0:
        for col in var_df.loc[date_or_id==1]['index'].values.tolist():
            if len(train[col].value_counts()) == len(train):
                if train[col].min() < 1900 or train[col].max() > 2050:
                    var_df.loc[var_df['index']==col,'id_col'] = 1
                else:
                    try:
                        pd.to_datetime(train[col],infer_datetime_format=True)
                        var_df.loc[var_df['index']==col,'date_time'] = 1
                    except:
                        var_df.loc[var_df['index']==col,'id_col'] = 1
            else:
                if train[col].min() < 1900 or train[col].max() > 2050:
                    if col not in num_bool_vars:
                        var_df.loc[var_df['index']==col,'int'] = 1
                else:
                    try:
                        pd.to_datetime(train[col],infer_datetime_format=True)
                        var_df.loc[var_df['index']==col,'date_time'] = 1
                    except:
                        if col not in num_bool_vars:
                            var_df.loc[var_df['index']==col,'int'] = 1
    else:
        pass

    int_vars = list(var_df[(var_df['int'] ==1)]['index'])
    date_vars = list(var_df[(var_df['date_time'] == 1)]['index'])
    id_vars = list(var_df[(var_df['id_col'] == 1)]['index'])
    sum_all_cols['int_vars'] = int_vars
    copy_date_vars = copy.deepcopy(date_vars)

    for date_var in copy_date_vars:
        try:
            pd.to_datetime(train[date_var],infer_datetime_format=True)
        except:
            cols_delete.append(date_var)
            date_vars.remove(date_var)

    sum_all_cols['date_vars'] = date_vars
    sum_all_cols['id_vars'] = id_vars
    sum_all_cols['cols_delete'] = cols_delete
    
    var_df['numeric'] = 0
    float_or_cat = var_df.apply(lambda x: 1 if x['type_of_column'] in ['float16',
                            'float32','float64'] else 0,
                                        axis=1)
    
    if len(var_df.loc[float_or_cat == 1]) > 0:
        for col in var_df.loc[float_or_cat == 1]['index'].values.tolist():
            if len(train[col].value_counts()) > 2 and len(train[col].value_counts()
                ) <= float_limit and len(train[col].value_counts()) <= len(train):
                var_df.loc[var_df['index']==col,'cat'] = 1
            else:
                if col not in (num_bool_vars + factor_vars):
                    var_df.loc[var_df['index']==col,'numeric'] = 1

    cat_vars = list(var_df[(var_df['cat'] ==1)]['index'])
    continuous_vars = list(var_df[(var_df['numeric'] ==1)]['index'])
    cat_vars_copy = copy.deepcopy(factor_vars)

    for cat in cat_vars_copy:
        if df_preds[cat].dtype==float:
            continuous_vars.append(cat)
            factor_vars.remove(cat)
            var_df.loc[var_df['index']==cat,'dcat'] = 0
            var_df.loc[var_df['index']==cat,'numeric'] = 1
        elif len(df_preds[cat].value_counts()) == df_preds.shape[0]:
            id_vars.append(cat)
            factor_vars.remove(cat)
            var_df.loc[var_df['index']==cat,'dcat'] = 0
            var_df.loc[var_df['index']==cat,'id_col'] = 1

    sum_all_cols['factor_vars'] = factor_vars
    cat_vars_copy = copy.deepcopy(cat_vars)

    for cat in cat_vars_copy:
        if df_preds[cat].dtype==float:
            continuous_vars.append(cat)
            cat_vars.remove(cat)
            var_df.loc[var_df['index']==cat,'cat'] = 0
            var_df.loc[var_df['index']==cat,'numeric'] = 1
        elif len(df_preds[cat].value_counts()) == df_preds.shape[0]:
            id_vars.append(cat)
            cat_vars.remove(cat)
            var_df.loc[var_df['index']==cat,'cat'] = 0
            var_df.loc[var_df['index']==cat,'id_col'] = 1

    sum_all_cols['cat_vars'] = cat_vars
    sum_all_cols['continuous_vars'] = continuous_vars
    sum_all_cols['id_vars'] = id_vars
    
    var_dict_sum = dict(zip(var_df.values[:,0], var_df.values[:,2:].sum(1)))

    for col, sumval in var_dict_sum.items():
        if sumval == 0:
            print('%s of type=%s is not classified' %(col,train[col].dtype))
        elif sumval > 1:
            print('%s of type=%s is classified into more then one type' %(col,train[col].dtype))
        else:
            pass
    
    copy_discretes = copy.deepcopy(discrete_string_vars)

    for each_discrete in copy_discretes:
        if train[each_discrete].nunique() >= 1000:
            nlp_vars.append(each_discrete)
            discrete_string_vars.remove(each_discrete)
        elif train[each_discrete].nunique() > 100 and train[each_discrete].nunique() < 1000:
            pass
        else:
            cat_vars.append(each_discrete)
            discrete_string_vars.remove(each_discrete)

    sum_all_cols['discrete_string_vars'] =  discrete_string_vars
    sum_all_cols['cat_vars'] = cat_vars
    sum_all_cols['nlp_vars'] = nlp_vars

    if verbose == 1:
        print("    Number of Numeric Columns = ", len(continuous_vars))
        print("    Number of Integer-Categorical Columns = ", len(int_vars))
        print("    Number of String-Categorical Columns = ", len(cat_vars))
        print("    Number of Factor-Categorical Columns = ", len(factor_vars))
        print("    Number of String-Boolean Columns = ", len(string_bool_vars))
        print("    Number of Numeric-Boolean Columns = ", len(num_bool_vars))
        print("    Number of Discrete String Columns = ", len(discrete_string_vars))
        print("    Number of NLP String Columns = ", len(nlp_vars))
        print("    Number of Date Time Columns = ", len(date_vars))
        print("    Number of ID Columns = ", len(id_vars))
        print("    Number of Columns to Delete = ", len(cols_delete))
    if verbose >= 2:
        print('  Printing upto %d columns (max) in each category:' %max_cols_to_print)
        print("    Numeric Columns : %s" %continuous_vars[:max_cols_to_print])
        print("    Integer-Categorical Columns: %s" %int_vars[:max_cols_to_print])
        print("    String-Categorical Columns: %s" %cat_vars[:max_cols_to_print])
        print("    Factor-Categorical Columns: %s" %factor_vars[:max_cols_to_print])
        print("    String-Boolean Columns: %s" %string_bool_vars[:max_cols_to_print])
        print("    Numeric-Boolean Columns: %s" %num_bool_vars[:max_cols_to_print])
        print("    Discrete String Columns: %s" %discrete_string_vars[:max_cols_to_print])
        print("    NLP text Columns: %s" %nlp_vars[:max_cols_to_print])
        print("    Date Time Columns: %s" %date_vars[:max_cols_to_print])
        print("    ID Columns: %s" %id_vars[:max_cols_to_print])
        print("    Columns that will not be considered in modeling: %s" %cols_delete[:max_cols_to_print])

    len_sum_all_cols = reduce(add,[len(v) for v in sum_all_cols.values()])

    if len_sum_all_cols == orig_cols_total:
        print('    All variables classified into correct types.' )
    else:
        print('No of columns classified %d does not match %d total cols. Continuing...' %(
                   len_sum_all_cols, orig_cols_total))
        ls = sum_all_cols.values()
        flat_list = [item for sublist in ls for item in sublist]

        if len(left_subtract(list(train),flat_list)) == 0:
            print(' Missing columns = None')
        else:
            print(' Missing columns = %s' %left_subtract(list(train),flat_list))

    return sum_all_cols

def left_subtract(l1,l2):
    lst = []

    for i in l1:
        if i not in l2:
            lst.append(i)
    return lst

def EDA_find_remove_columns_with_infinity(df, remove=False, verbose=0):
    nums = df.select_dtypes(include='number').columns.tolist()
    dfx = df[nums]
    sum_rows = np.isinf(dfx).values.sum()
    add_cols =  list(dfx.columns.to_series()[np.isinf(dfx).any()])

    if sum_rows > 0:
        if verbose > 0:
            print('    there are %d rows and %d columns with infinity in them...' %(sum_rows,len(add_cols)))
        if remove:
            nocols = [x for x in df.columns if x not in add_cols]
            if verbose > 0:
                print("    Shape of dataset before %s and after %s removing columns with infinity" %(df.shape,(df[nocols].shape,)))
            return df[nocols]
        else:
            
            return add_cols
    else:
        return add_cols

module_type = 'Running' if  __name__ == "__main__" else 'Imported'
version_number =  '1.02'
