def actors_top_1_by_decades(df: pd.DataFrame):
    grouped_df = (
        df.groupby(["cuts", "person_name"], observed=True)
        .size()
        .reset_index(name="total_film_acteurs")
        .sort_values(by="total_film_acteurs")
    )

    top_acteurs_decennie = (
        grouped_df.groupby("cuts", observed=True)
        .apply(lambda x: x.nlargest(1, "total_film_acteurs"))
        .reset_index(drop=True)
    )

    decennies = top_acteurs_decennie["cuts"]
    noms_acteurs = top_acteurs_decennie["person_name"]
    nombre_films = top_acteurs_decennie["total_film_acteurs"]
