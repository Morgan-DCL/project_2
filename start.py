import asyncio

from api_tmdb import main as machine_learning
from api_update_tmdb import main as tmdb
# from get_dataframes import GetDataframes
# from get_dataframes_full_polars import GetDataframes
from tools import import_config

from datetime import datetime
time = {}

for name in ["pandas", "polars"]:
    start = datetime.now()
    def get_all():
        config = import_config()
        # asyncio.run(tmdb())
        # asyncio.run(machine_learning())
        if name == "pandas":
            from get_dataframes import GetDataframes
            all_for_one = GetDataframes(config)
            all_for_one.get_all_dataframes()
        else:
            from get_dataframes_full_polars import GetDataframes
            all_for_one = GetDataframes(config)
            all_for_one.get_all_dataframes()
    get_all()
    end = datetime.now()
    time[name] = f"Temps écoulé -> {end - start}"

print(time)