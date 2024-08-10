import logging
import os
import asyncio
import time
import aiohttp
import gc
from collections import defaultdict
from typing import List, Dict
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from ultralytics import YOLO
from models.solicitacao_model import SolicitacaoModel
from models.usuario_model import UsuarioModel
from schemas.solicitacao_schema import SolicitacaoCreate, PolesRequest
from core.deps import get_session, get_current_user
from motor.motor_asyncio import AsyncIOMotorClient
from decouple import config

logging.getLogger('ultralytics').setLevel(logging.ERROR)


router = APIRouter()
semaphore = asyncio.Semaphore(5)

modelos = [YOLO(os.path.join('ia', file)) for file in os.listdir('ia') if file.endswith('.pt')]
modelos_nome = [file.replace('model_', '').replace('.pt', '') for file in os.listdir('ia') if file.endswith('.pt')]
model_names_map = {
    'Poste_Distribuidora': 'Poste',
    'Trafo': 'Equipamentos',
    'Transformador': 'Equipamentos',
    'Chave': 'Equipamentos',
    'Seccionalizador': 'Equipamentos',
    'Religador': 'Equipamentos',
    'Regulador': 'Equipamentos',
    'Para_raio': 'Equipamentos',
    'Capacitor': 'Equipamentos',
    'UM': 'UM',
    'BT': 'BT', 
    'BT_Convencional': 'BT',
    'BT_Multiplexada': 'BT',
    'MT_Space': 'MT',
    'MT_Convencional': 'MT',
    'IP': 'IP',
    'MT': 'MT',
    'Pan_Poste_Distribuidora': 'Poste',
    'Esp_Equipamentos': 'Equipamentos',
    'Pan_UM': 'UM',
    'Esp_BT': 'BT',
    'Pan_BT': 'BT',
    'Esp_IP': 'IP',
    'Pan_MT': 'MT',
    'Esp_MT': 'MT',
    'Pan_IP': 'IP'
}


def is_model_selected(model, models_selected):
    for name in model.names.values():
        if model_names_map.get(name) in models_selected:
            return True
    return False


async def save_to_mongodb(data: Dict, solicitacao_id: int):
    client = AsyncIOMotorClient(config('MONGO_URL'))
    db = client["test"]  
    collection = db["solicitacoes"]
    result = await collection.insert_one({"solicitacao_id": solicitacao_id, "data": data})
    client.close()  
    return result.inserted_id


async def predict_model(model, images):
    async with semaphore:
        try:
            return await asyncio.to_thread(model.predict, images)
        except:
            return await asyncio.to_thread(model.predict, images)


async def check_image_exists(url: str) -> bool:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.head(url) as response:
                return response.status == 200
    except Exception as e:
        logging.error(f"Erro ao verificar URL {url}: {e}")
        return False


async def process_pole(pole, models_selected) -> Dict:
    valid_images = []
    photo_ids = []
    for photo in pole.Photos:
        if await check_image_exists(photo.URL):
            valid_images.append(photo.URL)
            photo_ids.append(photo.PhotoId)
        else:
            logging.warning(f"Imagem inválida: {photo.URL}")
    if not valid_images:
        return {"PoleId": pole.PoleId, "Photos": []}
    images = valid_images
    
    filtered_models = [model for model in modelos if is_model_selected(model, models_selected)]
    filtered_models_names = [model for model in modelos_nome if model_names_map.get(model) in models_selected]

    tasks = [predict_model(model, images) for model in filtered_models]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    pole_results = []
    for idx, (photo_id, image) in enumerate(zip(photo_ids, images)):
        pole_result = {}

        for i, modelo_nome in enumerate(filtered_models_names):
            res = results[i][idx]

            if isinstance(res, Exception):
                pole_result[modelo_nome] = "Não foi possível abrir esta imagem"
            else:
                if len(res.names) == 1:
                    max_conf = round(max((box.conf.item() for box in res.boxes), default=0), 3)
                    pole_result[modelo_nome] = (max_conf > 0, max_conf)
                else:
                    class_confidences = {res.names[int(box.cls)]: (round(box.conf.item(), 3) > 0, round(box.conf.item(), 3)) for box in res.boxes}
                    pole_result[modelo_nome] = class_confidences

        pole_results.append({
            "PhotoId": photo_id,
            "URL": image,
            "Resultado": pole_result
        })

    output = {"PoleId": pole.PoleId, "Photos": pole_results}
    gc.collect()
    return output


async def detect_objects(request: PolesRequest, solicitacao_id: int):
    batch_size = 5
    pole_results = []

    for i in range(0, len(request.Poles), batch_size):
        batch = request.Poles[i:i + batch_size]
        pole_tasks = [process_pole(pole, request.Models) for pole in batch]
        batch_results = await asyncio.gather(*pole_tasks)
        pole_results.extend(batch_results)
        gc.collect()

    summarized_results = summarize_results(pole_results)

    response = {str(solicitacao_id): summarized_results}
    inserted_id = await save_to_mongodb(response, solicitacao_id)
    print(f"Resultado salvo no MongoDB com ID: {inserted_id}")

    if request.webhook_url:
        async with aiohttp.ClientSession() as session:
            async with session.post(str(request.webhook_url), json=response) as resp:
                if resp.status != 200:
                    print(f"Falha ao enviar resultado para o webhook: {resp.status}")
                else:
                    print("Resultado enviado para o webhook")

    gc.collect()
    return response


def summarize_results(pole_results):
    summary_data = defaultdict(lambda: defaultdict(float))                                   
    summary_spec = defaultdict(lambda: defaultdict(float))

    for pole in pole_results:
        pole_id = pole["PoleId"]
        for photo in pole["Photos"]:
            resultado = photo["Resultado"]
            for key, value in resultado.items():   
                key_type = key.split('_')[1]
                
                if len(value) >= 1:
                    if isinstance(value, tuple):
                        summary_data[pole_id][key_type] = max(summary_data[pole_id][key_type], value[1])
                    elif isinstance(value, dict):
                        key_spec = next(iter(value))
                        summary_spec[pole_id][key_spec] = max(summary_spec[pole_id][key_spec], value[key_spec][1])
                        summary_data[pole_id][key_type] = max(summary_data[pole_id][key_type], value[next(iter(value))][1])
                    else:
                        print('Tipo não reconhecido.')
            
    summarized_results = [
        {
            "Poste_ID": pole_id,
            "Resultado": summary,
            "Especificidades": summary_spec[pole_id]
        }
        for pole_id, summary in summary_data.items()
    ]

    return summarized_results



async def start_detection(solicitacao_id: int, db: AsyncSession, poles_request: PolesRequest):
    start_time = time.time()
    async with db as session:
        try:
            detection_results = await detect_objects(request=poles_request, solicitacao_id=solicitacao_id)
            await update_status(solicitacao_id=solicitacao_id, status='Concluído', db=session)
            end_time = time.time()
            print(f"- Solicitação {solicitacao_id} concluída em: {end_time - start_time:.2f} segundos")
            gc.collect()
            return detection_results
        except Exception as e:
            await update_status(solicitacao_id=solicitacao_id, status='Falhou', db=session)
            end_time = time.time()
            print(f"- Solicitação {solicitacao_id} concluída em: {end_time - start_time:.2f} segundos")
            gc.collect()
            raise e


@router.post("/", response_model=SolicitacaoCreate)
async def criar_solicitacao(poles_request: PolesRequest, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_session), usuario_logado: UsuarioModel = Depends(get_current_user)):
    start_time = time.time()
    total_poles = len(poles_request.Poles)
    total_photos = sum(len(pole.Photos) for pole in poles_request.Poles)
    if total_poles > 100:
        raise HTTPException(status_code=400, detail="Número de postes não pode ser maior que 100.")
    if total_photos > 1000:
        raise HTTPException(status_code=400, detail="Número de fotos não pode ser maior que 1000.")

    nova_solicitacao: SolicitacaoModel = SolicitacaoModel(status="Em andamento", postes=total_poles, imagens=total_photos, usuario_id=usuario_logado.id)
    async with db as session:
        session.add(nova_solicitacao)
        await session.commit()
        await session.refresh(nova_solicitacao)

        background_tasks.add_task(start_detection, nova_solicitacao.id, session, poles_request)

        end_time = time.time()
        print(f"- Solicitação criada em: {end_time - start_time:.2f} segundos")
        return {"id": nova_solicitacao.id, "status": nova_solicitacao.status, "postes": nova_solicitacao.postes, "imagens": nova_solicitacao.imagens}


@router.get("/status/{solicitacao_id}")#, response_model=SolicitacaoCreate)
async def status(solicitacao_id: int, db: AsyncSession = Depends(get_session), usuario_logado: UsuarioModel = Depends(get_current_user)):
    async with db as session:
        query = select(SolicitacaoModel).filter(SolicitacaoModel.id == solicitacao_id)
        result = await session.execute(query)
        solicitacao = result.scalars().unique().one_or_none()

        if not solicitacao:
            raise HTTPException(status_code=404, detail="Nenhuma solicitação encontrada")

        if solicitacao.status == "Concluído":
            client = AsyncIOMotorClient(config('MONGO_URL'))
            db = client["test"]  
            collection = db["solicitacoes"] 
            document = await collection.find_one({"solicitacao_id": solicitacao_id})
            client.close() 
            if not document:
                raise HTTPException(status_code=404, detail="Arquivo de resultado não encontrado") 
            return document["data"]

        return [solicitacao]


async def update_status(solicitacao_id: int, status: str, db: AsyncSession):
    async with db as session:
        query = select(SolicitacaoModel).filter(SolicitacaoModel.id == solicitacao_id)
        result = await session.execute(query)
        solicitacao = result.scalars().unique().one_or_none()
        if solicitacao:
            solicitacao.status = status
            await session.commit()
            await session.refresh(solicitacao)
