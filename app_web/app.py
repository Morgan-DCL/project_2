import pandas as pd
from flask import Flask, render_template, request
from fuzzywuzzy import process

app = Flask(__name__)

df = pd.read_parquet("clean_datasets/machine_learning.parquet")

print(df.titre_clean)

TEMPLATE = "test.html"


@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        search_query = request.form["search"]
        best_match = process.extractOne(search_query, df["titre_str"].values)

        infos = df[df["titre_str"] == best_match[0]].iloc[0]

        result = {
            "titre_genres": infos["titre_genres"],
            "titre_str": infos["titre_str"],
            "titre_id": infos["titre_id"],
            "tmdb_id": infos["tmdb_id"],
            "actors": infos["actors"],
            "directors": infos["director"],
            "popularity": infos["popularity"],
            "youtube": infos["youtube"],
            "image": infos["image"],
        }
        result["youtube"] = result["youtube"].replace("watch?v=", "embed/")
        result["youtube"] = (
            result["youtube"]
            + "?autoplay=1&autohide=2&border=0&wmode=opaque&enablejsapi=1&modestbranding=1&controls=0&showinfo=1&mute=1"
        )

    return render_template(TEMPLATE, result=result)


# if __name__ == '__main__':
#     app.run(debug=True)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         search_term = request.form['search']
#         # Recherche dans votre DataFrame
#         result = df[df['titre_str'].str.contains(search_term)].iloc[0]
#         # Modification de la colonne 'youtube' pour l'intégration
#         result['youtube'] = result['youtube'].replace('watch?v=', 'embed/')
#         return render_template('test.html', result=result)
#     return render_template('test.html')


if __name__ == "__main__":
    app.run(debug=True)
