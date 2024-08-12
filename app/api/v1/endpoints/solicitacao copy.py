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
import psutil

logging.getLogger('ultralytics').setLevel(logging.ERROR)

router = APIRouter()
start_detection_semaphore = asyncio.Semaphore(1)
predict_model_semaphore = asyncio.Semaphore(20)

modelos, modelos_nome = zip(*[(YOLO(os.path.join('ia', file)), file.replace('model_', '').replace('.pt', '')) for file in sorted(os.listdir('ia')) if file.endswith('.pt')])
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


def print_memory_usage(label: str):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"[{label}] Memory Usage: {mem_info.rss / (1024 * 1024):.2f} MB")


def is_model_selected(model, models_selected):
    #print_memory_usage("is_model_selected - start")
    for name in model.names.values():
        if model_names_map.get(name) in models_selected:
            #print_memory_usage("is_model_selected - end")
            return True
    #print_memory_usage("is_model_selected - end")
    return False


async def save_to_mongodb(data: Dict, solicitacao_id: int):
    #print_memory_usage("save_to_mongodb - start")
    async with AsyncIOMotorClient(config('MONGO_URL')) as client:
        db = client["test"]  
        collection = db["solicitacoes"]
        result = await collection.insert_one({"solicitacao_id": solicitacao_id, "data": data})
        #print_memory_usage("save_to_mongodb - end")
        return result.inserted_id


async def predict_model(model, images):
    #print_memory_usage("predict_model - start")
    async with predict_model_semaphore:
        try:
            result = await asyncio.to_thread(model.predict, images)
            #print_memory_usage("predict_model - end")
            return result
        except Exception as e:
            logging.error(f"Erro na predição: {e}")
            #print_memory_usage("predict_model - end")
            return e


async def check_image_exists(url: str) -> bool:
    #print_memory_usage("check_image_exists - start")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.head(url) as response:
                #print_memory_usage("check_image_exists - end")
                return response.status == 200
    except Exception as e:
        logging.error(f"Erro ao verificar URL {url}: {e}")
        #print_memory_usage("check_image_exists - error")
        return False


async def process_pole(pole, models_selected) -> Dict:
    #print_memory_usage("process_pole - start")
    valid_images = []
    photo_ids = []
    for photo in pole.Photos:
        if await check_image_exists(photo.URL):
            valid_images.append(photo.URL)
            photo_ids.append(photo.PhotoId)
        else:
            logging.warning(f"Imagem inválida: {photo.URL}")
    if not valid_images:
        #print_memory_usage("process_pole - end")
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
    del filtered_models, results, images, valid_images, photo_ids, pole_results
    gc.collect()
    #print_memory_usage("process_pole - end")
    return output


async def detect_objects(request: PolesRequest, solicitacao_id: int):
    #print_memory_usage("detect_objects - start")
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
    print(f"--- Resultado salvo com ID: {inserted_id}")

    if request.webhook_url:
        async with aiohttp.ClientSession() as session:
            async with session.post(str(request.webhook_url), json=response) as resp:
                if resp.status != 200:
                    print(f"Falha ao enviar resultado para o webhook: {resp.status}")
                else:
                    print("Resultado enviado para o webhook")

    gc.collect()
    #print_memory_usage("detect_objects - end")
    return response


def summarize_results(pole_results):
    #print_memory_usage("summarize_results - start")
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

    del summary_data, summary_spec
    gc.collect()
    #print_memory_usage("summarize_results - end")
    return summarized_results


async def start_detection(solicitacao_id: int, db: AsyncSession, poles_request: PolesRequest):
    #print_memory_usage("start_detection - start")
    async with start_detection_semaphore:
        start_time = time.time()
        async with db as session:
            try:
                detection_results = await detect_objects(request=poles_request, solicitacao_id=solicitacao_id)
                await update_status(solicitacao_id=solicitacao_id, status='Concluído', db=session)
                end_time = time.time()
                print(f"- Solicitação {solicitacao_id} concluída em: {end_time - start_time:.2f} segundos")
                gc.collect()
                #print_memory_usage("start_detection - end")
                return detection_results
            except Exception as e:
                await update_status(solicitacao_id=solicitacao_id, status='Falhou', db=session)
                end_time = time.time()
                print(f"- Solicitação {solicitacao_id} concluída em: {end_time - start_time:.2f} segundos")
                gc.collect()
                #print_memory_usage("start_detection - end")
                raise e
            finally:
                del detection_results, poles_request
                gc.collect()
                #print_memory_usage("start_detection - end")


@router.post("/", response_model=SolicitacaoCreate)
async def criar_solicitacao(poles_request: PolesRequest, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_session), usuario_logado: UsuarioModel = Depends(get_current_user)):
    #print_memory_usage("criar_solicitacao - start")
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
        gc.collect()
        print(f"- Solicitação {nova_solicitacao.id} criada em: {end_time - start_time:.2f} segundos")
        #print_memory_usage("criar_solicitacao - end")
        return {"id": nova_solicitacao.id, "status": nova_solicitacao.status, "postes": nova_solicitacao.postes, "imagens": nova_solicitacao.imagens}


@router.get("/status/{solicitacao_id}")#, response_model=SolicitacaoCreate)
async def status(solicitacao_id: int, db: AsyncSession = Depends(get_session), usuario_logado: UsuarioModel = Depends(get_current_user)):
    #print_memory_usage("status - start")
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
            #print_memory_usage("status - end")
            return document["data"]

        #print_memory_usage("status - end")
        return [solicitacao]


async def update_status(solicitacao_id: int, status: str, db: AsyncSession):
    #print_memory_usage("update_status - start")
    async with db as session:
        query = select(SolicitacaoModel).filter(SolicitacaoModel.id == solicitacao_id)
        result = await session.execute(query)
        solicitacao = result.scalars().unique().one_or_none()
        if solicitacao:
            solicitacao.status = status
            await session.commit()
            await session.refresh(solicitacao)
    #print_memory_usage("update_status - end")