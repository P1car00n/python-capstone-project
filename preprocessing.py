from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

input_directory_path = Path(input('Input directory: '))
output_file_path = input('Output directory: ') + '/' + 'fsae_mi_2013-2017'

dfs = (pd.read_csv(path_to_file, encoding='utf8')
       for path_to_file in input_directory_path.iterdir())

res_csv = pd.concat(
    dfs, keys=[
        '2013', '2014', '2015', '2016', '2017'], names=[
            'Year', 'Row ID'])
res_csv.to_parquet(output_file_path + '.parquet', index=True)
