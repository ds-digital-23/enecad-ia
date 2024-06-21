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


logging.getLogger('ultralytics').setLevel(logging.ERROR)

router = APIRouter()
semaphore = asyncio.Semaphore(100)


async def process_image_with_models(image: str) -> Dict[str, Dict[str, float]]:
    async with semaphore:
        tasks_image = [asyncio.to_thread(loaded_models[model_name]["model"].predict, [image]) for model_name in modelos]
        results = await asyncio.gather(*tasks_image)

        combined_results = {}
        for model_name, result in zip(modelos, results):
            detected = any(len(res.boxes) > 0 for res in result[0])
            max_conf = round(max((box.conf.item() for box in result[0].boxes), default=0), 3)
            combined_results[loaded_models[model_name]["nome"]] = {
                    "detected": detected,
                    "max_confidence": max_conf
                }
        return combined_results


async def process_batch_images(images: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    tasks = [process_image_with_models(image) for image in images]
    results = await asyncio.gather(*tasks)
    return dict(zip(images, results))


async def process_images(images: List[str], photo_ids: List[int]) -> List[Resultado]:
    start_time = time.time()
    detection_results = await process_batch_images(images)
    resultados = []
    for photo_id, image in zip(photo_ids, images):
        detection_result = detection_results[image]
        resultados.append(Resultado(PhotoId=photo_id, URL=image, Resultado=detection_result))
    end_time = time.time()
    print(f"process_images function took {end_time - start_time:.2f} seconds")
    return resultados


async def process_pole(pole):
    images = [photo.URL for photo in pole.Photos]
    photo_ids = [photo.PhotoId for photo in pole.Photos]
    results = await process_images(images, photo_ids)
    pole_results = {"PoleId": pole.PoleId, "Photos": [result.model_dump() for result in results]}
    return pole_results


async def detect_objects(request: PolesRequest, solicitacao_id: int):
    start_time = time.time()
    pole_tasks = [process_pole(pole) for pole in request.Poles]
    pole_results = await asyncio.gather(*pole_tasks)
    
    os.makedirs('results', exist_ok=True)
    response = {str(solicitacao_id): pole_results}
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


async def start_detection(solicitacao_id: int, db: AsyncSession, poles_request: PolesRequest):
    start_time = time.time()
    global modelos
    modelos = list(loaded_models.keys())
    print('Modelos carregados em solicitacao', modelos)

    async with db as session:
        try:    
            detection_results = await detect_objects(request=poles_request, solicitacao_id=solicitacao_id)
            await update_status(solicitacao_id=solicitacao_id, status='Concluído', db=session)
            end_time = time.time()
            print(f"start_detection function took {end_time - start_time:.2f} seconds")
            return detection_results
        
        except Exception as e:
            await update_status(solicitacao_id=solicitacao_id, status='Falhou', db=session)
            end_time = time.time()
            print(f"start_detection function took {end_time - start_time:.2f} seconds")
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

        background_tasks.add_task(start_detection, nova_solicitacao.id, session, poles_request)

        end_time = time.time()
        print(f"criar_solicitacao function took {end_time - start_time:.2f} seconds")
        return {"id": nova_solicitacao.id, "status": nova_solicitacao.status, "postes": nova_solicitacao.postes, "imagens": nova_solicitacao.imagens}


@router.get("/status/{solicitacao_id}", response_model=List[SolicitacaoCreate])
async def obter_solicitacao(solicitacao_id: int, db: AsyncSession = Depends(get_session), usuario_logado: UsuarioModel = Depends(get_current_user)):
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
            return FileResponse(path=response_file_path, filename=f"solicitacao_{solicitacao_id}.json", media_type='application/json')

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
