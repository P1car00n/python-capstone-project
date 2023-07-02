from pathlib import Path

import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import MinMaxScaler

input_directory_path = Path(input('Input directory: '))
output_file_path = input('Output directory: ') + '/' + 'fsae_mi_2013-2017'

dfs = (pd.read_csv(path_to_file, encoding='utf8')
       for path_to_file in input_directory_path.iterdir())

res_df = pd.concat(
    dfs, keys=[
        '2013', '2014', '2015', '2016', '2017'], names=[
            'Year', 'Row ID'])
res_df.dropna(inplace=True)
res_df.drop(columns='Car Num', inplace=True)

hasher = FeatureHasher(n_features=40, input_type='string')
hashed_array = hasher.fit_transform(
    [[team] for team in res_df['Team']]).toarray()
hashed_data = pd.DataFrame(
    hashed_array,
    index=res_df.index,
    columns=[
        'hashed_' +
        str(i) for i in range(
            hashed_array.shape[1])])

res_df = res_df.join(hashed_data)
res_df.drop(columns='Team', inplace=True)

res_df.to_parquet(output_file_path + '.parquet', index=True)
