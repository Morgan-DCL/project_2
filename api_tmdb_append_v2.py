import asyncio
import aiohttp
import ast
import pandas as pd
import json
from datetime import datetime, timedelta
from tools import logging, import_config, make_filepath, color


async def fetch(ss, url, params):
    while True:
        async with ss.get(url, params=params) as rsp:
            if rsp.status == 429:
                logging.error("Attention Rate Limit API")
                await asyncio.sleep(10)
                continue
            return await rsp.json()

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
        "vote_average.gte": "5.5",
        "vote_count.gte": "1000",
        "with_runtime.gte": "63",
        "with_runtime.lte": "230",
        "without_genres": "Documentary",
        "page": 1
    }
    rsp = await fetch(
        ss, base_url, params=params
    )
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

    return [
        r["results"] for
        r in rsps if
        r and "results" in r
    ]

async def fetch_movies_ids(
    ss,
    config: dict,
    base_url: str,
    api_key: str,
    language: str,
):
    list_id_tmdb = set()
    start_date = datetime(config["movies_years"], 1, 1)
    end_date = datetime.now()
    step = timedelta(days=30)
    logging.info("Fetch all movies...")
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
        list_id_tmdb.update(m['id'] for mb in movies for m in mb)
        start_date = segment_end + timedelta(days=1)

    with open("testtthhhht.json", "w") as fp:
        json.dump(list(list_id_tmdb), fp, indent=1)
    return list(list_id_tmdb)


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
        "append_to_response": "keywords,credits"
    }

    base_url = "https://api.themoviedb.org/3/movie/"
    url = f"{base_url}{TMdb_id}"

    async with ss.get(url, params=params) as rsp:
        data = rsp.json()
        return await data


async def main():
    config = import_config()
    # with open("testtthhhht.json", "r") as fp:
    #     tmdb_id_list = json.load(fp)
    # tmdb_id_list = tmdb_id_list
    async with aiohttp.ClientSession() as ss:
        logging.info("Fetching TMdb ids...")
        tmdb_id_list = await fetch_movies_ids(
            ss,
            config,
            config["base_url"],
            config["tmdb_api_key"],
            config["language"]
        )

        logging.info("Creating TMdb Dataframe...")
        taches = []
        for id in tmdb_id_list:
            tache = asyncio.create_task(
                get_movie_details(ss, id, config["tmdb_api_key"], config["language"])
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
        try:
            full = []
            for data in datas:
                if "imdb_id" not in data or not data["imdb_id"]:
                    logging.error(color(data, "red"))
                    continue
                else:
                    for k, c, v in cc:
                        data[k] = [k[v] for k in data[c]]

                    data["keywords"] = [
                        n["name"] for n in data["keywords"]["keywords"][:10]
                    ]
                    data["actors"] = [
                        n["name"] for n in data["credits"]["cast"] if
                        n["known_for_department"] == "Acting" and
                        n["order"] <= 4
                    ]
                    data["director"] = [
                        n["name"] for n in data["credits"]["crew"] if
                        n["job"] == "Director"
                    ]

                    data.pop("credits")
                    data.pop("homepage")
                    data.pop("belongs_to_collection")
                    data.pop("production_companies")
                    # data["release_date"] = data["release_date"][:4]
                    full.append(data)
        except KeyError as e:
            print(e)
            pass

    df = pd.DataFrame(full)
    df["release_date"] = pd.to_datetime(df["release_date"])
    df.to_csv("TESSSST.csv", index=False)
    logging.info("Cleaning...")
    # df = df[~df["imdb_id"].duplicated(keep="last")]
    df.reset_index(drop="index", inplace=True)
    logging.info("Saving updated TMdb dataframe...")
    base_ = make_filepath(config["clean_df_path"])
    df.to_parquet(f"{base_}/tmdb_updated_append.parquet")
    return df


if __name__ == "__main__":
    asyncio.run(main())



