import pandas as pd
import os
from pandas_profiling import ProfileReport
import seaborn as sns

"""
Reference
https://www.kaggle.com/code/mateuszk013/icr-eda-balanced-learning-with-lgbm-xgb

"""

df_train = pd.read_csv('./icr-identify-age-related-conditions/train.csv')
df_test = pd.read_csv('./icr-identify-age-related-conditions/test.csv')
df_geek = pd.read_csv('./icr-identify-age-related-conditions/greeks.csv')

ls_dataset = [df_train, df_test]

def run_pandas_profiling_report():
    # Create the ProfileReport objects
    profile_train = ProfileReport(df_train, title="Training Set Profiling Report")
    profile_test = ProfileReport(df_test, title="Test Set Profiling Report")
    profile_greek = ProfileReport(df_geek, title="Greeks Set Profiling Report")

    # Save the reports as HTML files
    path_report = './pandas_profiling_reports'
    os.mkdir(path_report) if os.path.exists('./pandas_profiling_reports') else 0
    profile_train.to_file("train_report.html")
    profile_test.to_file("test_report.html")
    profile_greek.to_file("geek_report.html")
    print('Report Generated')

def glance_at_dataset(df):
    for col in df.columns:
        if col == 'Id': continue
        print(f'{col}: {df[col].dtype}')
        if df[col].dtype == 'object':
            print(df[col].value_counts())

for data in ls_dataset:
    print(glance_at_dataset(data))

def preprocess_dataset(df):
    df['EJ'] = df['EJ'].replace({'A':0, 'B': 1})
    return df

## 061723 :(
def get_correlation_with_class(df):
    correlations = df_train.corr()['Class'].sort_values(ascending=False)
    return correlations

corr_train = get_correlation_with_class(df_train)
print(corr_train)


from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import pearsonr
import pandas as pd
import numpy as np


exit()
def exhaustive_feature_engineering(df, target_col):
    features = df.drop(columns=[target_col]).columns
    results = []

    for i in range(len(features)):
        for j in range(i+1, len(features)):
            # perform interaction and polynomial feature creation
            interaction_feature = df[features[i]] * df[features[j]]
            polynomial_feature = df[features[i]] ** 2
            interaction_feature_name = f"{features[i]}_X_{features[j]}"
            polynomial_feature_name = f"{features[i]}^2"

            for new_feature, new_feature_name in zip([interaction_feature, polynomial_feature],
                                                     [interaction_feature_name, polynomial_feature_name]):
                # compute the mutual information with the target
                mi = mutual_info_classif(new_feature.values.reshape(-1, 1), df[target_col])[0]

                # compute the absolute Pearson correlation with the target
                pearson_corr = abs(pearsonr(new_feature, df[target_col])[0])

                # compute the feature importance
                rf = RandomForestClassifier(n_estimators=10)
                rf.fit(df[[features[i], features[j]]], df[target_col])
                feat_imp = rf.feature_importances_

                results.append([new_feature_name, mi, pearson_corr, feat_imp[0], feat_imp[1]])

    # Create a DataFrame from results
    df_results = pd.DataFrame(results, columns=['Feature', 'Mutual_Info', 'Pearson_Corr', 'Feat_Imp_1', 'Feat_Imp_2'])
    return df_results

result = exhaustive_feature_engineering(df_train, df_train['Class'])
print(result)