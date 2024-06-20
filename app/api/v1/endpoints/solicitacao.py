import logging
import os
import asyncio
import time
import json
import aiohttp
import gc
from typing import List, Dict
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from ultralytics import YOLO
from models.solicitacao_model import SolicitacaoModel
from models.usuario_model import UsuarioModel
from schemas.solicitacao_schema import SolicitacaoCreate, PolesRequest, Resultado
from core.deps import get_session, get_current_user
from models_loader import loaded_models


router = APIRouter()
semaphore = asyncio.Semaphore(20)
model = None  # Variável global para armazenar o primeiro modelo


async def process_batch_images(images: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    async with semaphore:
        start_time = time.time()

        detection_results = []
        model_start_time = time.time()
        results = await asyncio.to_thread(model.predict, images, stream=True)
        model_end_time = time.time()
        print(f"Model {model} processed images in {model_end_time - model_start_time:.2f} seconds")
        detection_results.append(("default_model", results))

        end_time = time.time()
        print(f"process_batch_images function took {end_time - start_time:.2f} seconds")
        logging.info(f"Processed {len(images)} images with 1 model in {end_time - start_time} seconds")

    combined_results = {}
    for model_name, results in detection_results:
        model_results = {}
        for image, result in zip(images, results):
            image_start_time = time.time()
            max_conf = round(max((res.conf.item() for res in result.boxes), default=0), 3)
            image_end_time = time.time()
            print(f"Image {image} processed in {image_end_time - image_start_time:.2f} seconds")
            model_results[image] = {
                "detected": any(len(res.boxes) > 0 for res in result),
                "max_confidence": max_conf
            }
        combined_results[loaded_models[model_name]["nome"]] = model_results

    gc.collect()
    return combined_results


async def process_images(images: List[str], photo_ids: List[int]) -> List[Resultado]:
    start_time = time.time()
    detection_results = await process_batch_images(images)
    end_time = time.time()
    print(f"process_images function took {end_time - start_time:.2f} seconds")
    logging.info(f"Processed batch of {len(images)} images in {end_time - start_time} seconds")

    resultados = []
    for photo_id, image in zip(photo_ids, images):
        detection_result = {model: {"detected": detection_results[model][image]["detected"], "max_confidence": detection_results[model][image]["max_confidence"]} for model in detection_results}
        resultados.append(Resultado(PhotoId=photo_id, URL=image, Resultado=detection_result))

    return resultados


async def detect_objects(request: PolesRequest, solicitacao_id: int):
    start_time = time.time()
    response = {solicitacao_id: []}
    batch_size = 20
    for pole in request.Poles:
        images = [photo.URL for photo in pole.Photos]
        photo_ids = [photo.PhotoId for photo in pole.Photos]
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_photo_ids = photo_ids[i:i+batch_size]
            results = await process_images(batch_images, batch_photo_ids)
            pole_results = {"PoleId": pole.PoleId, "Photos": [result.model_dump() for result in results]}
            response[solicitacao_id].append(pole_results)

    os.makedirs('results', exist_ok=True)
    response_file_path = os.path.join('results', f"solicitacao_{solicitacao_id}.json")
    with open(response_file_path, 'w') as response_file:
        json.dump(response, response_file, indent=4)

    if request.webhook_url:
        async with aiohttp.ClientSession() as session:
            async with session.post(request.webhook_url, json=response) as resp:
                if resp.status != 200:
                    logging.error(f"Failed to post results to webhook: {resp.status}")

    end_time = time.time()
    print(f"detect_objects function took {end_time - start_time:.2f} seconds")
    return response


@router.get("/obter_solicitacao/{solicitacao_id}", response_model=List[SolicitacaoCreate])
async def obter_solicitacao(solicitacao_id: int, db: AsyncSession = Depends(get_session), usuario_logado: UsuarioModel = Depends(get_current_user)):
    start_time = time.time()
    async with db as session:
        query = select(SolicitacaoModel).filter(SolicitacaoModel.id == solicitacao_id)
        result = await session.execute(query)
        solicitacao = result.scalars().unique().one_or_none()

        if not solicitacao:
            raise HTTPException(status_code=404, detail="Nenhuma solicitação encontrada")

        if solicitacao.status == "Concluído":
            response_file_path = os.path.join('results', f"solicitacao_{solicitacao_id}.json")
            if not os.path.exists(response_file_path):
                raise HTTPException(status_code=404, detail="Arquivo de resultado não encontrado")
            end_time = time.time()
            print(f"obter_solicitacao function took {end_time - start_time:.2f} seconds")
            return FileResponse(path=response_file_path, filename=f"solicitacao_{solicitacao_id}.json", media_type='application/json')

        end_time = time.time()
        print(f"obter_solicitacao function took {end_time - start_time:.2f} seconds")
        return [solicitacao]


async def update_status(solicitacao_id: int, status: str, db: AsyncSession):
    start_time = time.time()
    async with db as session:
        query = select(SolicitacaoModel).filter(SolicitacaoModel.id == solicitacao_id)
        result = await session.execute(query)
        solicitacao = result.scalars().unique().one_or_none()
        if solicitacao:
            solicitacao.status = status
            await session.commit()
            await session.refresh(solicitacao)
    end_time = time.time()
    print(f"update_status function took {end_time - start_time:.2f} seconds")


async def trigger_model_and_detection_tasks(solicitacao_id: int, db: AsyncSession, poles_request: PolesRequest):
    global model  # Declare as global to modify the global variable
    start_time = time.time()
    async with db as session:
        try:
            modelos = list(loaded_models.keys())
            model = loaded_models[modelos[0]]["model"]  # Assign the first model to the global variable
            detection_results = await detect_objects(request=poles_request, solicitacao_id=solicitacao_id)
            await update_status(solicitacao_id=solicitacao_id, status='Concluído', db=session)
            end_time = time.time()
            print(f"trigger_model_and_detection_tasks function took {end_time - start_time:.2f} seconds")
            return detection_results
        except Exception as e:
            await update_status(solicitacao_id=solicitacao_id, status='Falhou', db=session)
            end_time = time.time()
            print(f"trigger_model_and_detection_tasks function took {end_time - start_time:.2f} seconds")
            raise e


@router.post("/", response_model=SolicitacaoCreate)
async def criar_solicitacao(poles_request: PolesRequest, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_session), usuario_logado: UsuarioModel = Depends(get_current_user)):
    start_time = time.time()
    total_poles = len(poles_request.Poles)
    total_photos = sum(len(pole.Photos) for pole in poles_request.Poles)

    if total_poles > 100:
        raise HTTPException(status_code=400, detail="Número de postes não pode ser maior que 100.")

    nova_solicitacao: SolicitacaoModel = SolicitacaoModel(status="Em andamento", postes=total_poles, imagens=total_photos, usuario_id=usuario_logado.id)

    async with db as session:
        session.add(nova_solicitacao)
        await session.commit()
        await session.refresh(nova_solicitacao)

        background_tasks.add_task(trigger_model_and_detection_tasks, nova_solicitacao.id, session, poles_request)

        end_time = time.time()
        print(f"criar_solicitacao function took {end_time - start_time:.2f} seconds")
        return {"id": nova_solicitacao.id, "status": nova_solicitacao.status, "postes": nova_solicitacao.postes, "imagens": nova_solicitacao.imagens}
