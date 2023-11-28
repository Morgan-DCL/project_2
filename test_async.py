import asyncio

import aiohttp
import pandas as pd

from tools import import_config, logging

# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.neighbors import NearestNeighbors
# from st_click_detector import click_detector


# import streamlit as st
# import streamlit.components.v1 as components


async def fetch_infos(
    ss: object,
    TMdb_id: int,
    config: dict,
):
    params = {
        "api_key": config["tmdb_api_key"],
        "include_adult": "False",
        "language": config["language"],
        "append_to_response": "combined_credits",
    }
    base_url = "https://api.themoviedb.org/3/person/"
    url = f"{base_url}{TMdb_id}"
    async with ss.get(url, params=params) as rsp:
        return await rsp.json()


async def fetch_persons_bio(
    config: dict, people_list: list, director: bool = False
) -> list:
    url_image = "https://image.tmdb.org/t/p/w300_and_h450_bestv2"
    async with aiohttp.ClientSession() as ss:
        taches = []
        for id in people_list:
            tache = asyncio.create_task(fetch_infos(ss, id, config))
            taches.append(tache)
        datas = await asyncio.gather(*taches)
        full = []
        for data in datas:
            data["image"] = f"{url_image}{data['profile_path']}"
            # 99: Documentaire, 16: Animation, 10402: Musique
            exclude = [99, 10402] if director else [99, 16, 10402]
            if director:
                top_credits = sorted(
                    (
                        n
                        for n in data["combined_credits"]["crew"]
                        if n["media_type"] == "movie"
                        and n["job"] == "Director"
                        and all(
                            genre not in n["genre_ids"]
                            for genre in exclude
                        )
                    ),
                    key=lambda x: (
                        -x["popularity"],
                        -x["vote_average"],
                        -x["vote_count"],
                    ),
                )[:8]
            else:
                top_credits = sorted(
                    (
                        n
                        for n in data["combined_credits"]["cast"]
                        if n["media_type"] == "movie"
                        and n["order"] <= 3
                        and all(
                            genre not in n["genre_ids"]
                            for genre in exclude
                        )
                    ),
                    key=lambda x: (
                        -x["popularity"],
                        -x["vote_average"],
                        -x["vote_count"],
                    ),
                )[:8]
            data["top_5"] = [n["title"] for n in top_credits]
            data["top_5_images"] = [
                f"{url_image}{n['poster_path']}" for n in top_credits
            ]
            data["top_5_movies_ids"] = [n["id"] for n in top_credits]
            to_pop = (
                "adult",
                "also_known_as",
                "gender",
                "homepage",
                "profile_path",
                "combined_credits",
                "known_for_department",
            )
            for tp in to_pop:
                data.pop(tp)
            full.append(data)
    return full


config = import_config()

df = pd.read_parquet("streamlit/datasets/site_web.parquet")


def get_actors_dict(df: pd.DataFrame) -> dict:
    actors_dict = {}
    for actors, ids in zip(df.actors, df.actors_ids):
        actors_list = actors.split(", ")
        actor_id_pairs = zip(actors_list, ids)
        actors_dict.update(actor_id_pairs)
    return actors_dict


def get_directors_dict(df: pd.DataFrame) -> dict:
    directors_dict = {}
    for directors, ids in zip(df.director, df.director_ids):
        directors_list = directors.split(", ")
        directors_id_pairs = zip(directors_list, ids)
        directors_dict.update(directors_id_pairs)
    return directors_dict


from datetime import datetime

start = datetime.now()
dfs = df[df["titre_str"] == "Avatar"]

name_actors = [a for a in get_actors_dict(dfs).keys()]
actors_list = [a for a in get_actors_dict(dfs).values()]

name_directors = [a for a in get_directors_dict(dfs).keys()]
directors_list = [a for a in get_directors_dict(dfs).values()]

actors = asyncio.run(fetch_persons_bio(config, actors_list))
directors = asyncio.run(fetch_persons_bio(config, directors_list, True))
import pprint

pprint.pprint([n for n in actors])
pprint.pprint(directors)
end = datetime.now()
print(f"Temps écoulé -> {end - start}")
