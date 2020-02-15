import pandas as pd
data_info = pd.read_csv('../DATA/lending_club_info.csv', index_col = 'LoanStatNew')

print(data_info.loc['revol_util']['Description'])
