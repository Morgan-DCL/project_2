import asyncio
import aiohttp
import ast
import pandas as pd
import json
from datetime import datetime, timedelta
from tools import logging, import_config, make_filepath


async def fetch(ss, url, params):
    async with ss.get(url, params=params) as rsp:
        return await rsp.json()

async def get_all_movies(
    ss,
    base_url: str,
    api_key: str,
    language: str,
    start_date: str,
    end_date: str,
):
    # date = datetime.strftime(datetime.now(), '%Y-%m-%d')
    params = {
        "api_key": api_key,
        'include_adult': "False",
        "language": language,
        "sort_by": "primary_release_date.desc",
        "primary_release_date.gte": start_date,
        "primary_release_date.lte": end_date,
        "page": 1
    }
    rsp = await fetch(
        ss, base_url, params=params
    )
    total_pages = (
        rsp['total_pages'] if
        rsp['total_pages'] <= 500 else 500
    )
    taches = [
        asyncio.ensure_future(
            fetch(
                ss,
                base_url,
                {
                    **params,
                    "page": page
                }
            )
        ) for page in range(1, total_pages + 1)
    ]
    rsps = await asyncio.gather(*taches)

    return [
        r['results'] for
        r in rsps if
        r and 'results' in r
    ]

async def fetch_movies_ids(
    base_url: str,
    api_key: str,
    language: str,
):
    all_movies_df = pd.DataFrame()
    start_date = datetime(2023, 5, 1)
    end_date = datetime.now()
    step = timedelta(days=30)
    logging.info("Fetch all movies...")
    async with aiohttp.ClientSession() as ss:
        while start_date < end_date:
            segment_end = min(start_date + step, end_date)
            movies = await get_all_movies(
                ss,
                base_url,
                api_key,
                language,
                start_date.strftime('%Y-%m-%d'),
                segment_end.strftime('%Y-%m-%d'),
            )
            segment_df = pd.DataFrame(sum(movies, []))
            all_movies_df = pd.concat(
                [all_movies_df, segment_df], ignore_index=True
            )
            start_date = segment_end + timedelta(days=1)

    logging.info("Droping duplicated TMdb IDs...")
    all_movies_df.drop_duplicates(
        subset=["id"], keep='first', inplace=True
    )
    list_id_tmdb = all_movies_df.id.to_list()
    with open("movies_ids.json", "w") as fp:
        json.dump(list_id_tmdb, fp, indent=1)
    return list_id_tmdb


async def get_movie_details(
    ss,
    TMdb_id: int,
    api_key: str,
    language: str,
):
    """Récupère les détails d'un film par son ID TMDB."""
    base_url = "https://api.themoviedb.org/3/movie/"
    url = f"{base_url}{TMdb_id}?api_key={api_key}&language={language}"
    async with ss.get(url) as rsp:
        return await rsp.json()


def clean_df(
    df: pd.DataFrame
):
    tt = (
        ("genres", "genres", "name"),
        ("spoken_languages", "spoken_languages", "iso_639_1"),
        ("production_companies_name", "production_companies", "name"),
        ("production_countries", "production_countries", "iso_3166_1"),
    )
    for t in tt:
        df[t[0]] = df[t[1]].apply(
            lambda x: [i[t[2]] for i in x]
        )
    col_to_drop = [
        "belongs_to_collection",
        "production_companies"
    ]
    add_col = [
        "status_code",
        "status_message",
        "success"
    ]
    col_to_drop.extend(c for c in add_col if c in df.columns)
    df.drop(columns=col_to_drop, inplace=True)
    return df


async def main():
    config = import_config()
    logging.info("Fetching TMdb ids...")
    tmdb_id_list = await fetch_movies_ids(
        config["base_url"],
        config["tmdb_api_key"],
        config["language"]
    )
    logging.info("Creating TMdb Dataframe...")
    async with aiohttp.ClientSession() as ss:
        taches = [
            asyncio.create_task(
                get_movie_details(
                    ss,
                    id,
                    config["tmdb_api_key"],
                    config["language"],
                )
            ) for id in tmdb_id_list
        ]
        movies_details = await asyncio.gather(*taches)

    pandas_df = pd.DataFrame(movies_details)
    logging.info("Droping NaN IMdb IDs...")
    pandas_df.dropna(subset=["imdb_id"], axis=0, inplace=True)
    logging.info("Cleaning...")
    pandas_df = clean_df(pandas_df)
    logging.info("Saving updated TMdb dataframe...")
    base_ = make_filepath(config["clean_df_path"])
    pandas_df.to_parquet(f"{base_}/tmdb_update.parquet")
    return pandas_df


if __name__ == "__main__":
    asyncio.run(main())