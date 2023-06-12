import pandas as pd
import os
from pandas_profiling import ProfileReport

df_train = pd.read_csv('./icr-identify-age-related-conditions/train.csv')
df_test = pd.read_csv('./icr-identify-age-related-conditions/test.csv')
df_geek = pd.read_csv('./icr-identify-age-related-conditions/greeks.csv')

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


