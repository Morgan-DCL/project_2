import os
import json
import requests
import gzip
import shutil
from tools import logging

def get_files(
    url: str,
    filename: str
):
    """
    Télécharge un fichier à partir d'une URL spécifiée et l'enregistre localement.

    Paramètres
    ----------
    url : str
        L'URL du fichier à télécharger.
    filename : str
        Le nom du fichier sous lequel le contenu téléchargé doit être enregistré.

    Notes
    -----
    Cette fonction utilise les modules 'requests' et 'shutil'
    pour effectuer le téléchargement et l'enregistrement du fichier.
    Elle ne renvoie rien mais enregistre le fichier téléchargé dans
    le répertoire de travail actuel.
    """

    rsp = requests.get(url, stream=True)
    with open(filename, 'wb') as dl:
        shutil.copyfileobj(rsp.raw, dl)


def extract_gz(
    gz_path: str,
    dest_path: str
):
    """
    Extrait le contenu d'un fichier zip et le
    sauvegarde dans un autre fichier.

    Paramètres
    ----------
    gz_path : str
        Le chemin vers le fichier gzip à extraire.
    dest_path : str
        Le chemin où le contenu extrait doit être sauvegardé.

    Notes
    -----
    Cette fonction n'a pas de valeur de retour.
    Elle écrit simplement le contenu extrait
    dans le fichier de destination spécifié.
    """

    with gzip.open(gz_path, 'rb') as gzip_:
        with open(dest_path, 'wb') as final:
            shutil.copyfileobj(gzip_, final)


def downloader(
    folder_name: str = "movies_datasets"
):
    """
    Fonction principale pour télécharger et
    extraire des ensembles de données de films.

    Cette fonction lit un fichier JSON contenant
    des liens vers des ensembles de données,
    télécharge l'ensembles des données dans un dossier spécifié,
    réalise l'extraction et supprime les fichiers compressés.

    Paramètres
    ----------
    folder_name : str, optionnel
        Le nom du dossier dans lequel les ensembles de données seront
        téléchargés et extraits. Par défaut, il s'agit de "movies_datasets".

    Notes
    -----
    Cette fonction ne renvoie rien. Elle télécharge,
    extrait et supprime des fichiers dans le système de fichiers local.
    """

    with open("datasets_link.json", 'r') as file:
        data = json.load(file)

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    for name, url in data["data_sets_tar"].items():
        logging.info(f"Téléchargement de {name}...")
        gz_file = os.path.join(folder_name, name + ".tsv.gz")
        tsv_file = os.path.join(folder_name, name + ".tsv")

        get_files(url, gz_file)
        logging.info(f"Extraction de {name}.tsv.gz...")
        extract_gz(gz_file, tsv_file)

    for del_gz in os.listdir(folder_name):
        if del_gz.endswith(".gz"):
            os.remove(os.path.join(folder_name, del_gz))


