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
from sqlalchemy import insert
from ultralytics import YOLO
from models.solicitacao_model import SolicitacaoModel
from models.arquivo_model import ArquivoModel
from models.usuario_model import UsuarioModel
from schemas.solicitacao_schema import SolicitacaoCreate, PolesRequest
from core.deps import get_session, get_current_user
from decouple import config



logging.getLogger('ultralytics').setLevel(logging.ERROR)

router = APIRouter()
start_detection_semaphore = asyncio.Semaphore(1)
predict_model_semaphore = asyncio.Semaphore(20)


modelos, modelos_nome = zip(*[(YOLO(os.path.join('ia', file)), file.replace('model_', '').replace('.pt', '')) for file in sorted(os.listdir('ia')) if file.endswith('.pt')])
model_names_map = {
    'Poste': 'Poste', 'Poste_Distribuidora': 'Poste', 'Poste_Prefeitura': 'Poste',
    'Transformador': 'Equipamentos', 'Chave': 'Equipamentos', 'Seccionalizador': 'Equipamentos', 'Religador': 'Equipamentos', 'Regulador': 'Equipamentos', 'Para_raio': 'Equipamentos', 'Capacitor': 'Equipamentos',
    'UM': 'UM',
    'BT': 'BT', 'BT_Convencional': 'BT','BT_Multiplexada': 'BT',
    'MT': 'MT','MT_Space': 'MT','MT_Convencional': 'MT',
    'IP': 'IP','IP_Lampada_Acesa': 'IP',
    'Esp_BT': 'BT','Esp_Equipamentos': 'Equipamentos','Esp_IP': 'IP','Esp_IP_Lampada_Acesa': 'IP','Esp_MT': 'MT','Esp_Poste': 'Poste','Esp_UM': 'UM',
    'Pan_BT': 'BT','Pan_IP': 'IP','Pan_MT': 'MT','Pan_Poste': 'Poste','Pan_UM': 'UM'  
}


def is_model_selected(model, models_selected):
    for name in model.names.values():
        if model_names_map.get(name) in models_selected:
            return True
    return False


async def save_to_postgresql(json: Dict, solicitacao_id: int, db: AsyncSession):
    query = insert(ArquivoModel).values(solicitacao_id=solicitacao_id, json=json)
    await db.execute(query)
    await db.commit()
    return solicitacao_id


async def predict_model(model, images):
    async with predict_model_semaphore:
        try:
            result = await asyncio.to_thread(model.predict, images)
            return result
        except Exception as e:
            logging.error(f"Erro na predição: {e}")
            return e


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
                    if max_conf > 0.0:
                        pole_result[modelo_nome] = (max_conf > 0, max_conf)
                else:
                    class_confidences = {res.names[int(box.cls)]: (round(box.conf.item(), 3) > 0, round(box.conf.item(), 3)) for box in res.boxes}
                    if class_confidences:
                        pole_result[modelo_nome] = class_confidences

        pole_results.append({
            "PhotoId": photo_id,
            "URL": image,
            "Resultado": pole_result
        })

    output = {"PoleId": pole.PoleId, "Photos": pole_results}
    del tasks, filtered_models, results, images, valid_images, photo_ids, pole_results, res
    gc.collect()
    return output


async def detect_objects(request: PolesRequest, solicitacao_id: int, db: AsyncSession):
    batch_size = 10
    pole_results = []

    for i in range(0, len(request.Poles), batch_size):
        batch = request.Poles[i:i + batch_size]
        pole_tasks = [process_pole(pole, request.Models) for pole in batch]
        batch_results = await asyncio.gather(*pole_tasks)
        pole_results.extend(batch_results)
        gc.collect()

    summarized_results = summarize_results(pole_results)

    response = {str(solicitacao_id): summarized_results}
    inserted_id = await save_to_postgresql(response, solicitacao_id, db)

    if request.webhook_url:
        async with aiohttp.ClientSession() as session:
            async with session.post(str(request.webhook_url), json=response) as resp:
                if resp.status != 200:
                    print(f"Falha ao enviar resultado para o webhook: {resp.status}")
                else:
                    print("Resultado enviado para o webhook")

    del summarized_results, pole_results, batch_results, pole_tasks, batch
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
                        for sub_key, sub_value in value.items():
                            if sub_key != 'UM':
                                summary_spec[pole_id][sub_key] = max(summary_spec[pole_id][sub_key], sub_value[1])
                            summary_data[pole_id][key_type] = max(summary_data[pole_id][key_type], sub_value[1])
                    else:
                        print('Tipo não reconhecido.')

    final_summary_spec = {}
    for pole_id, spec in summary_spec.items():
        final_summary_spec[pole_id] = spec.copy()
        for category in ["BT", "MT", "Poste"]:
            category_items = {k: v for k, v in spec.items() if category in k}
            if category_items:
                max_item = max(category_items, key=category_items.get)
                for item in category_items:
                    if item != max_item:
                        final_summary_spec[pole_id].pop(item, None)

    final_summary_spec[pole_id] = dict(sorted(final_summary_spec[pole_id].items()))

    summarized_results = [
        {
            "Poste_ID": pole_id,
            "Resultado": summary,
            "Especificidades": final_summary_spec.get(pole_id, {})
        }
        for pole_id, summary in summary_data.items()
    ]

    del summary_data, summary_spec, pole_results
    gc.collect()
    return summarized_results


async def start_detection(solicitacao_id: int, db: AsyncSession, poles_request: PolesRequest):
    async with start_detection_semaphore:
        start_time = time.time()
        async with db as session:
            try:
                detection_results = await detect_objects(request=poles_request, solicitacao_id=solicitacao_id, db=session)
                await update_status(solicitacao_id=solicitacao_id, status='Concluído', db=session)
                end_time = time.time()
                gc.collect()
                print(f"- Solicitação {solicitacao_id} concluída em: {end_time - start_time:.2f} segundos")
                return detection_results
            except Exception as e:
                detection_results = await update_status(solicitacao_id=solicitacao_id, status='Falhou', db=session)
                end_time = time.time()
                gc.collect()
                print(f"- Solicitação {solicitacao_id} concluída em: {end_time - start_time:.2f} segundos")
                raise e


@router.post("/", response_model=SolicitacaoCreate)
async def criar_solicitacao(poles_request: PolesRequest, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_session), usuario_logado: UsuarioModel = Depends(get_current_user)):
    for pole in poles_request.Poles:
        seen_urls = set()
        unique_photos = []
        for photo in pole.Photos:
            if photo.URL not in seen_urls:
                unique_photos.append(photo)
                seen_urls.add(photo.URL)
        pole.Photos = unique_photos
    
    start_time = time.time()
    total_poles = len(poles_request.Poles)
    total_photos = sum(len(pole.Photos) for pole in poles_request.Poles)
    if total_photos > 800:
        raise HTTPException(status_code=400, detail="Número de fotos não pode ser maior que 800.")

    nova_solicitacao: SolicitacaoModel = SolicitacaoModel(status="Em andamento", postes=total_poles, imagens=total_photos, usuario_id=usuario_logado.id)
    async with db as session:
        session.add(nova_solicitacao)
        await session.commit()
        await session.refresh(nova_solicitacao)

        background_tasks.add_task(start_detection, nova_solicitacao.id, session, poles_request)

        end_time = time.time()
        gc.collect()
        print(f"- Solicitação {nova_solicitacao.id} criada em: {end_time - start_time:.2f} segundos")
        return {"id": nova_solicitacao.id, "status": nova_solicitacao.status, "postes": nova_solicitacao.postes, "imagens": nova_solicitacao.imagens}


@router.get("/status/{solicitacao_id}")
async def status(solicitacao_id: int, db: AsyncSession = Depends(get_session), usuario_logado: UsuarioModel = Depends(get_current_user)):
    async with db as session:
        query = select(SolicitacaoModel).filter(SolicitacaoModel.id == solicitacao_id)
        result = await session.execute(query)
        solicitacao = result.scalars().unique().one_or_none()

        if not solicitacao:
            raise HTTPException(status_code=404, detail="Nenhuma solicitação encontrada")

        if solicitacao.status == "Concluído":
            query = select(ArquivoModel).filter(ArquivoModel.solicitacao_id == solicitacao_id)
            result = await session.execute(query)
            solicitacao_data = result.scalar_one_or_none()
            if not solicitacao_data:
                raise HTTPException(status_code=404, detail="Arquivo de resultado não encontrado")
            return solicitacao_data.json

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

