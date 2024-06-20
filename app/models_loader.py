import os
import asyncio
import gdown
from ultralytics import YOLO


models_to_download = {
    "model_ip_v1.3.pt": {"url": "https://drive.google.com/uc?id=1HhZTpDf_3XH_DBNnew6_kTmV7YDnQJkB", "nome": "IP"}
}

async def download_model(file_name, url):
    output = os.path.join('ia', file_name)
    print(output)
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

async def download_all_models():
    tasks = [download_model(file_name, details["url"]) for file_name, details in models_to_download.items()]
    await asyncio.gather(*tasks)

async def load_models():
    model_directory = 'ia' 

    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    await download_all_models()

    return "Modelos foram baixados e estão na pasta 'ia'."
