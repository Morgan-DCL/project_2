import gzip
import json
import os
import shutil

import requests

from tools import logging, get_tsv_files


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


def download_extract(
    config: dict,
    folder_name: str,
    need_file: str = None
):
    for name, url in config["data_sets_tar"].items():
        if need_file is not None and name != need_file:
            continue

        logging.info(f"Téléchargement de {name}...")
        gz_file = os.path.join(folder_name, name + ".tsv.gz")
        tsv_file = os.path.join(folder_name, name + ".tsv")

        get_files(url, gz_file)
        logging.info(f"Extraction de {name}.tsv.gz...")
        extract_gz(gz_file, tsv_file)

    for del_gz in os.listdir(folder_name):
        if del_gz.endswith(".gz"):
            os.remove(os.path.join(folder_name, del_gz))
    logging.info("Files ready to use!")


def downloader(
    config: dict,
):
    folder_name = config["download_path"]
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)

    data_sets_tsv = get_tsv_files(folder_name)
    data_sets_tsv.pop("imdb_full", None)

    if config["download"]:
        logging.info(f"Downloading all files...")
        download_extract(config, folder_name)
    else:
        miss_file = [
            n for n, path in data_sets_tsv.items()
            if not os.path.exists(path)
        ]
        if any(miss_file):
            logging.info(f"File {miss_file[0]} not found. Downloading...")
            for need in miss_file:
                download_extract(config, folder_name, need)
        else:
            logging.info(f"TSV files already exist.")
