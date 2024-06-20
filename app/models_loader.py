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

async def load_model():
    model_directory = 'ia' 

    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    await download_all_models()

    loaded_models = {}
    model_files = [f for f in os.listdir(model_directory) if f.endswith('.pt') and os.path.isfile(os.path.join(model_directory, f))]
    for model_file in model_files:
        model_path = os.path.join(model_directory, model_file)
        model_ia = await asyncio.to_thread(YOLO, model_path)
        loaded_models[model_file] = {"model": model_ia, "nome": models_to_download[model_file]["nome"]}
    
    print("Modelos carregados:", loaded_models)

    return loaded_models
