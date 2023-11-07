import asyncio
import aiohttp
import traceback
import pandas as pd
import json
import pprint

with open("testtt.json") as fp:
    small = json.load(fp)

with open("tmdb.json", "r") as fp:
    key = json.load(fp)

api_key = key["api_key"]

async def get_movie_details(
    session,
    TMdb_id: int,
    use_IMdb: bool = False,
    IMdb_id: str = ""
):
    """Récupère les détails d'un film par son ID TMDB."""
    if use_IMdb:
        base_url = "https://api.themoviedb.org/3/find/"
        url = f"{base_url}{IMdb_id}?api_key={api_key}&external_source=imdb_id&language=fr"
    else:
        base_url = "https://api.themoviedb.org/3/movie/"
        url = f"{base_url}{TMdb_id}?api_key={api_key}&language=fr"
    async with session.get(url) as response:
        return await response.json()

async def main():
    async with aiohttp.ClientSession() as ss:
        tasks = [asyncio.create_task(get_movie_details(ss, TMdb_id=id, use_IMdb=False)) for id in small]
        movies_details = await asyncio.gather(*tasks)

    pandas_df = pd.DataFrame(movies_details)
    pandas_df.dropna(subset=["imdb_id"], axis=0, inplace=True)
    pandas_df.to_csv("tmdb_update.csv", index=False)
    return pandas_df


from datetime import datetime
if __name__ == "__main__":
    start_ = datetime.now()
    asyncio.run(main())
    end = datetime.now()
    print(f"Temps ecoulé : {end - start_}")