
from urllib.parse import urlparse, parse_qs
import io
import os

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from patoolib import extract_archive

from osu_ml_difficulty import config
from import_replay_data import USERS_FILENAME
import pandas as pd

SHEET_ID = "1lpTCKNPoE2ikUN6liXjOcUgvnE7ePR8HDFD77YZe-rg"
CRED_PATH = "../secrets/replay_data_gdrive_credentials.json"
#SCOPE = ["https://googleapis.com/auth/drive"]


creds = ServiceAccountCredentials.from_json_keyfile_name(CRED_PATH)
gdrive_service = build("drive", "v3", credentials=creds)
gspread_service = gspread.authorize(creds)

def download_replays(username, url):
    extract_path = os.path.join(config.DOWNLOAD_PATH, username)
    if not os.path.exists(extract_path):
        fileId = parse_qs(urlparse(url).query)["id"][0]
        metadata = gdrive_service.files().get(fileId=fileId).execute()
        _, extension = os.path.splitext(metadata["name"])
        filepath = os.path.join(config.DOWNLOAD_PATH, username + extension)

        if not os.path.isfile(filepath):
            request = gdrive_service.files().get_media(fileId=fileId)

            with io.FileIO(filepath+".tmp", mode='wb') as fh:
                downloader = MediaIoBaseDownload(fh, request)

                done = False
                print(f"Downloading {filepath}")

                while done is False:
                    status, done = downloader.next_chunk()
                    if status:
                        print(f"Downloading {filepath}: {int(status.progress()*100)}%.")
            os.rename(filepath+".tmp", filepath)

        extract_archive(filepath, outdir=extract_path)



def download_responses():
    responses_csv = gdrive_service.files().export(fileId=SHEET_ID, mimeType="text/csv").execute()

    with open(USERS_FILENAME, "wb") as users_file:
        users_file.write(responses_csv)

    for _, row in pd.read_csv(USERS_FILENAME).iterrows():
        username = row["username"]
        url = row["replays_url"]
        download_replays(username, url)

if __name__ == "__main__":
    download_responses()