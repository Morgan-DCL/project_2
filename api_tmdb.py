import asyncio
from datetime import datetime, timedelta

import aiohttp
import pandas as pd

from tools import color, import_config, logging, make_filepath


async def fetch(ss, url, params):
    while True:
        async with ss.get(url, params=params) as rsp:
            if rsp.status == 429:
                logging.error("Attention Rate Limit API")
                await asyncio.sleep(10)
                continue
            return await rsp.json()


async def fetch_movies_ids(
    ss,
    config: dict,
    base_url: str,
    api_key: str,
    language: str,
):
    start_date = datetime(config["tmdb_date"], 1, 1)
    end_date = datetime.now()
    step = timedelta(days=30)
    logging.info("Fetch all movies...")
    list_id_tmdb = set()
    while start_date < end_date:
        segment_end = min(start_date + step, end_date)
        movies = await get_all_movies(
            ss,
            config,
            base_url,
            api_key,
            language,
            start_date.strftime("%Y-%m-%d"),
            segment_end.strftime("%Y-%m-%d"),
        )
        list_id_tmdb.update(m["id"] for mb in movies for m in mb)
        start_date = segment_end + timedelta(days=1)
    return list(list_id_tmdb)


async def get_all_movies(
    ss,
    config: dict,
    base_url: str,
    api_key: str,
    language: str,
    start_date: str,
    end_date: str,
):
    params = {
        "api_key": api_key,
        "include_adult": "False",
        "language": language,
        "sort_by": "primary_release_date.desc",
        "primary_release_date.gte": start_date,
        "primary_release_date.lte": end_date,
        "vote_average.gte": str(config["movies_rating_avg"]),
        "vote_count.gte": "750",
        "with_runtime.gte": str(config["movies_min_duration"]),
        "with_runtime.lte": str(config["movies_max_duration"]),
        "without_genres": "Documentary",
        "page": 1,
    }
    rsp = await fetch(ss, base_url, params=params)
    total_pages = min(rsp["total_pages"], 500)
    taches = []
    for page in range(1, total_pages + 1):
        taches.append(
            asyncio.ensure_future(
                fetch(ss, base_url, {**params, "page": page})
            )
        )
        await asyncio.sleep(0.02)
    rsps = await asyncio.gather(*taches)
    return [r["results"] for r in rsps if r and "results" in r]


async def get_movie_details(
    ss,
    TMdb_id: int,
    api_key: str,
    language: str,
):
    params = {
        "api_key": api_key,
        "include_adult": "False",
        "language": language,
        "append_to_response": "keywords,credits,videos",
    }

    base_url = "https://api.themoviedb.org/3/movie/"
    url = f"{base_url}{TMdb_id}"

    async with ss.get(url, params=params) as rsp:
        data = rsp.json()
        return await data


async def main():
    config = import_config("config/config.hjson")
    async with aiohttp.ClientSession() as ss:
        logging.info("Fetching TMdb ids...")
        tmdb_id_list = await fetch_movies_ids(
            ss,
            config,
            config["base_url"],
            config["tmdb_api_key"],
            config["language"],
        )

        logging.info("Creating TMdb Dataframe...")
        taches = []
        for id in tmdb_id_list:
            tache = asyncio.create_task(
                get_movie_details(
                    ss, id, config["tmdb_api_key"], config["language"]
                )
            )
            taches.append(tache)
            await asyncio.sleep(0.02)
        datas = await asyncio.gather(*taches)
        cc = [
            ("genres", "genres", "name"),
            ("spoken_languages", "spoken_languages", "iso_639_1"),
            ("production_companies_name", "production_companies", "name"),
            ("production_countries", "production_countries", "iso_3166_1"),
        ]
        keys_ = ["imdb_id", "poster_path", "videos"]
        try:
            full = []
            for data in datas:
                if any(key not in data or not data[key] for key in keys_):
                    logging.error(color(data, "red"))
                    continue

                for k, c, v in cc:
                    data[k] = [k[v] for k in data[c]]

                data["keywords"] = [
                    n["name"]
                    for n in data["keywords"]["keywords"][
                        : config["tmdb_max_keywords"]
                    ]
                ]
                data["actors"] = [
                    n["name"]
                    for n in data["credits"]["cast"]
                    if n["known_for_department"] == "Acting"
                    and n["order"] <= config["tmdb_max_actors"] - 1
                ]
                data["director"] = [
                    n["name"]
                    for n in data["credits"]["crew"]
                    if n["job"] == "Director"
                ]
                data["url"] = f"https://www.imdb.com/title/{data['imdb_id']}"
                data[
                    "image"
                ] = f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
                if data["videos"]["results"]:
                    data["youtube"] = [
                        f"https://www.youtube.com/watch?v={n['key']}"
                        for n in data["videos"]["results"]
                    ][0]
                else:
                    data["youtube"] = ""

                to_pop = [
                    "videos",
                    "video",
                    "credits",
                    "homepage",
                    "belongs_to_collection",
                    "adult",
                    "original_language",
                    "backdrop_path",
                    "spoken_languages",
                    "status",
                    "original_title",
                    "production_companies",
                    "poster_path",
                ]
                for tp in to_pop:
                    data.pop(tp)
                full.append(data)
        except KeyError as e:
            print(e)

    df = pd.DataFrame(full)
    df["release_date"] = pd.to_datetime(df["release_date"])
    logging.info("Cleaning...")
    df.reset_index(drop="index", inplace=True)
    logging.info("Saving updated TMdb dataframe...")
    base_ = make_filepath(config["clean_df_path"])
    base_ = base_.lstrip("../")
    df.to_parquet(f"{base_}/machine_learning.parquet")
    return df


# if __name__ == "__main__":
#     asyncio.run(main())
