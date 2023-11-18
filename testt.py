from get_dataframes_full_polars import GetDataframes
from tools import import_config
# from get_dataframes import GetDataframes
config = import_config()
test = GetDataframes(config)

# pl.read_parquet("clean_datasets/characters_test.parquet")
df = test.get_directors_dataframe() # shape: (13_012_610, 4)