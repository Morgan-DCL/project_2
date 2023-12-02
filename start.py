import asyncio

import nltk

from api_tmdb import main as machine_learning
from api_update_tmdb import main as tmdb
from get_dataframes import GetDataframes
from tools import import_config, logging

nltk.download("stopwords")
nltk.download("wordnet")
from datetime import datetime

start = datetime.now()
def get_all():
    config = import_config()
    asyncio.run(tmdb(config))
    logging.info("Start creating dataframe for Machine Learning...")
    asyncio.run(machine_learning(config))
    all_for_one = GetDataframes(config)
    all_for_one.get_all_dataframes()


get_all()
end = datetime.now()
print(f"Temps écoulé -> {end - start}")
