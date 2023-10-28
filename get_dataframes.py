import pandas as pd
pd.set_option('display.float_format', lambda x: f'{x :.2f}')
from cleaner import DataCleaner
from tools import (
    logging,
    create_persons_dataframe,
    create_actors_and_directors_dataframe,
)
clean = DataCleaner()


link = "movies_datasets/name_basics.tsv"
create_persons_dataframe(link)
logging.info("Actors and Directors dataframes..")
link_principals = "movies_datasets/title_principals.tsv"
create_actors_and_directors_dataframe(link_principals, True, True)