import logging
import os
import asyncio
import time
import aiohttp
import gc
from collections import defaultdict
from typing import Dict, List
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



logging.getLogger('ultralytics').setLevel(logging.ERROR)

router = APIRouter()
start_detection_semaphore = asyncio.Semaphore(1)
predict_model_semaphore = asyncio.Semaphore(24)


@router.post("/", response_model=SolicitacaoCreate)
async def criar_solicitacao(poles_request: PolesRequest, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_session), usuario_logado: UsuarioModel = Depends(get_current_user)):
    start_time = time.time()

    for pole in poles_request.Poles:
        pole.Photos = list({photo.URL: photo for photo in pole.Photos}.values())
    
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
        print(f"- Solicitação {nova_solicitacao.id} criada em: {end_time - start_time:.2f} segundos")
        return {"id": nova_solicitacao.id, "status": nova_solicitacao.status, "postes": nova_solicitacao.postes, "imagens": nova_solicitacao.imagens}


async def start_detection(solicitacao_id: int, db: AsyncSession, poles_request: PolesRequest):
    async with start_detection_semaphore:
        async with db as session:
            start_time = time.time()
            try:
                detection_results = await detect_objects(request=poles_request, solicitacao_id=solicitacao_id, db=session)
                await update_status(solicitacao_id=solicitacao_id, status='Concluído', db=session)
            except Exception as e:
                await update_status(solicitacao_id=solicitacao_id, status='Falhou', db=session)
                raise e
            finally:
                end_time = time.time()
                gc.collect()
                print(f"Tempo total de start_detection: {end_time - start_time:.2f} segundos")
                print(f"- Solicitação {solicitacao_id} concluída em: {end_time - start_time:.2f} segundos")
        return detection_results


async def detect_objects(request: PolesRequest, solicitacao_id: int, db: AsyncSession):
    start_time = time.time()
    modelos, modelos_nome = load_requested_models(request.Models)
    pole_results = []

    async with aiohttp.ClientSession() as session:
        # Coletar todas as URLs de uma vez
        all_urls = [photo.URL for pole in request.Poles for photo in pole.Photos]
        
        # Verificar a existência de todas as imagens de uma vez e obter apenas as válidas
        valid_urls = await check_images_exist(session, all_urls)

        # Filtrar as fotos válidas para cada poste
        valid_url_set = set(valid_urls)  # Para facilitar a verificação
        for pole in request.Poles:
            pole.Photos = [photo for photo in pole.Photos if photo.URL in valid_url_set]
            
            if not pole.Photos:
                continue 

        # Processar apenas postes válidos
        results = await asyncio.gather(*[process_pole(pole, modelos, modelos_nome) for pole in request.Poles if pole.Photos])
        pole_results.extend(results)

    summarized_results = await summarize_results(pole_results)
    response = {str(solicitacao_id): summarized_results}
    inserted_id = await save_to_postgresql(response, solicitacao_id, db)

    if request.webhook_url:
        async with aiohttp.ClientSession() as session:
            async with session.post(str(request.webhook_url), json=response) as resp:
                if resp.status != 200:
                    print(f"Falha ao enviar resultado para o webhook: {resp.status}")
                else:
                    print("Resultado enviado para o webhook")

    del summarized_results, pole_results
    end_time = time.time()
    print(f"Tempo total de detect_objects: {end_time - start_time:.2f} segundos")
    return response


def load_requested_models(models_selected):
    start_time = time.time()
    available_models = {
        file.replace('model_', '').replace('.pt', ''): file
        for file in sorted(os.listdir('ia'))
        if file.endswith('.pt')
    }

    modelos, modelos_nome = zip(*[
        (YOLO(os.path.join('ia', available_models[a_model])), a_model)
        for a_model in available_models
        for model_name in models_selected
        if model_name in a_model
    ])

    end_time = time.time()
    print(f"Tempo total de load_requested_models: {end_time - start_time:.2f} segundos")
    return list(modelos), list(modelos_nome)


async def check_images_exist(session: aiohttp.ClientSession, urls: List[str]) -> List[str]:
    start_time = time.time()
    tasks = []

    # Criar uma tarefa para cada URL
    for url in urls:
        tasks.append(session.head(url))

    # Executar todas as tarefas em paralelo
    responses = await asyncio.gather(*tasks)
    
    end_time = time.time()
    print(f"Tempo total de check_images_exist: {end_time - start_time:.2f} segundos")
    
    # Retornar apenas as URLs que são válidas (status 200)
    valid_urls = [url for url, response in zip(urls, responses) if response.status == 200]
    return valid_urls


async def process_pole(pole, modelos, modelos_nome) -> Dict:
    start_time = time.time()
    images = [photo.URL for photo in pole.Photos] 
    photo_ids = [photo.PhotoId for photo in pole.Photos]
    
    print('process_role: before', photo_ids)
    tasks = [predict_model(model, images) for model in modelos]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print('process_role: after', photo_ids)

    pole_results = []
    for idx, (photo_id, image) in enumerate(zip(photo_ids, images)):
        pole_result = {}

        for i, modelo_nome in enumerate(modelos_nome):
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
            "Resultado": pole_result,
        })

    output = {"PoleId": pole.PoleId, "Photos": pole_results}

    end_time = time.time()
    print('process_role: end', photo_ids)
    print(f"Tempo total de process_pole: {end_time - start_time:.2f} segundos")
    return output


async def summarize_results(pole_results):
    start_time = time.time()
    summary_data = defaultdict(lambda: defaultdict(float))
    summary_spec = defaultdict(lambda: defaultdict(float))
    photo_counts = defaultdict(lambda: defaultdict(int))

    for pole in pole_results:
        pole_id = pole["PoleId"]
        num_photos = len(pole["Photos"])
        print('summarized_results', pole_id, num_photos)

        for photo in pole["Photos"]:
            resultado = photo["Resultado"]
            image_url = photo["URL"]
            counted_key_types = set()

            for key, value in resultado.items():
                key_type = key.split('_')[1]
                
                if key_type not in counted_key_types:
                    photo_counts[pole_id][key_type] += 1
                    counted_key_types.add(key_type)
                if key.startswith("Esp"):
                    photo_counts[pole_id][key_type] += 2
                    counted_key_types.add(key_type)

                if isinstance(value, tuple):
                    summary_data[pole_id][key_type] = max(summary_data[pole_id][key_type], value[1])
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if sub_key != 'UM':
                            summary_spec[pole_id][sub_key] = max(summary_spec[pole_id][sub_key], sub_value[1])
                        summary_data[pole_id][key_type] = max(summary_data[pole_id][key_type], sub_value[1])

    summarized_results = []

    for pole_id in summary_data:
        final_data = {}
        for k, v in summary_data[pole_id].items():
            if photo_counts[pole_id][k] > 1 or num_photos == 1:
                final_data[k] = v
        
        final_spec = summary_spec.get(pole_id, {}).copy()

        for category in ["BT", "MT", "Poste"]:
            category_items = {k: v for k, v in final_spec.items() if category in k}
            if category_items:
                max_item = max(category_items, key=category_items.get)
                final_spec = {k: v for k, v in final_spec.items() if k == max_item or k not in category_items}

        summarized_results.append({
            "Poste_ID": pole_id,
            "Resultado": final_data,
            "Especificidades": final_spec
        })

    end_time = time.time()
    print(f"Tempo total de summarize_results: {end_time - start_time:.2f} segundos")
    return summarized_results


async def save_to_postgresql(json: Dict, solicitacao_id: int, db: AsyncSession):
    start_time = time.time()
    query = insert(ArquivoModel).values(solicitacao_id=solicitacao_id, json=json)
    await db.execute(query)
    await db.commit()
    end_time = time.time()
    print(f"Tempo total de save_to_postgresql: {end_time - start_time:.2f} segundos")
    return solicitacao_id


async def predict_model(model, images):
    async with predict_model_semaphore:
        try:
            start_time = time.time()
            result = await asyncio.to_thread(model.predict, images) 
            end_time = time.time()
            print(f"Tempo total de predict_model: {end_time - start_time:.2f} segundos")
            return result
        except Exception as e:
            logging.error(f"Erro na predição: {e}")
            return e


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
