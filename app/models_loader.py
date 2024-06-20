import os
import asyncio
import gdown
from ultralytics import YOLO

loaded_models = {}

# Link para download do modelo no Google Drive com atributo nome
models_to_download = {
    "model_ip_v1.3.pt": {"url": "https://drive.google.com/uc?id=1HhZTpDf_3XH_DBNnew6_kTmV7YDnQJkB", "nome": "IP"}
}

async def download_model(file_name, url):
    output = os.path.join('ia', file_name)
    print(f"Baixando o modelo para: {output}")
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

async def load_models():
    model_directory = 'ia' 

    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    await download_model("model_ip_v1.3.pt", models_to_download["model_ip_v1.3.pt"]["url"])

    model_path = os.path.join(model_directory, "model_ip_v1.3.pt")
    model_ia = await asyncio.to_thread(YOLO, model_path)
    loaded_models["model_ip_v1.3.pt"] = {"model": model_ia, "nome": models_to_download["model_ip_v1.3.pt"]["nome"]}
    print(os.listdir('ia'))
    print("Modelos carregados:", loaded_models)
    return loaded_models
