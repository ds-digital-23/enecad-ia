import logging
import os
import asyncio
import time
import json
import httpx
import gc
from typing import List, Dict
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from ultralytics import YOLO
from models.solicitacao_model import SolicitacaoModel
from models.usuario_model import UsuarioModel
from models.modelo_model import ModeloModel
from schemas.modelo_schema import ModeloResponse
from app.schemas.solicitacao_schema_conf import SolicitacaoCreate, PolesRequest, Resultado
from core.deps import get_session, get_current_user
from models_loader import loaded_models  # Importando os modelos carregados

router = APIRouter()
semaphore = asyncio.Semaphore(5)

async def get_model(model_description: str):
    for key, value in loaded_models.items():
        if value['description'] == model_description:
            return key
    raise ValueError(f"Modelo com descrição {model_description} não encontrado")

async def process_batch_images(images: List[str], modelos: List[str], confs: Dict[str, float]) -> Dict[str, Dict[str, Dict[str, float]]]:
    async with semaphore:
        detection_tasks = [get_model(modelo) for modelo in modelos]
        model_keys = await asyncio.gather(*detection_tasks)
        loaded_models_instances = [loaded_models[model_key]['model'] for model_key in model_keys]

        for model_instance, model_key in zip(loaded_models_instances, model_keys):
            model_instance.conf = confs[model_key]

        detection_results = await asyncio.gather(
            *[asyncio.to_thread(model.predict, images, stream=True) for model in loaded_models_instances]
        )

    combined_results = {}
    for model_key, results in zip(model_keys, detection_results):
        model_description = loaded_models[model_key]['description']
        model_results = {}
        for image, result in zip(images, results):
            if hasattr(result, 'boxes') and isinstance(result.boxes, (list, tuple)):
                max_conf = max((box.conf for box in result.boxes), default=0.0)
                detected = len(result.boxes) > 0
            else:
                max_conf = 0.0
                detected = False

            model_results[image] = {
                "detected": detected,
                "max_conf": max_conf
            }
        combined_results[model_description] = model_results

    gc.collect()

    return combined_results

async def process_images(images: List[str], modelos: List[str], photo_ids: List[int], confs: Dict[str, float]) -> List[Resultado]:
    start_time = time.time()
    detection_results = await process_batch_images(images, modelos, confs)
    end_time = time.time()
    logging.info(f"Processed batch of {len(images)} images in {end_time - start_time} seconds")

    resultados = []
    for photo_id, image in zip(photo_ids, images):
        detection_result = {
            model: {
                "detected": detection_results[model][image]["detected"],
                "max_conf": detection_results[model][image]["max_conf"]
            }
            for model in detection_results
        }
        resultados.append(Resultado(PhotoId=photo_id, URL=image, Resultado=detection_result))

    return resultados

async def detect_objects(request: PolesRequest, modelos: List[str], confs: Dict[str, float], solicitacao_id: int):
    response = {solicitacao_id: []}
    batch_size = 5 
    for pole in request.Poles:
        images = [photo.URL for photo in pole.Photos]
        photo_ids = [photo.PhotoId for photo in pole.Photos]
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_photo_ids = photo_ids[i:i+batch_size]
            results = await process_images(batch_images, modelos, batch_photo_ids, confs)
            pole_results = {"PoleId": pole.PoleId, "Photos": [result.model_dump() for result in results]}
            response[solicitacao_id].append(pole_results)

    os.makedirs('results', exist_ok=True)
    response_file_path = os.path.join('results', f"solicitacao_{solicitacao_id}.json")
    with open(response_file_path, 'w') as response_file:
        json.dump(response, response_file, indent=4)

    if request.webhook_url:
        async with httpx.AsyncClient() as client:
            await client.post(str(request.webhook_url), json=response)

    return response

@router.get("/obter_solicitacao/{solicitacao_id}", response_model=List[SolicitacaoCreate])
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

async def trigger_model_and_detection_tasks(solicitacao_id: int, db: AsyncSession, poles_request: PolesRequest):
    async with db as session:
        try:
            modelos = ["IP", "Lampada_Acesa"]
            confs = {
                "model_ip_v1.3.pt": 0.4,
                "model_la_v1.2.pt": 0.58
            }
            detection_results = await detect_objects(request=poles_request, modelos=modelos, confs=confs, solicitacao_id=solicitacao_id)
            await update_status(solicitacao_id=solicitacao_id, status='Concluído', db=session)
            return detection_results
        except Exception as e:
            await update_status(solicitacao_id=solicitacao_id, status='Falhou', db=session)
            raise e

@router.post("/", response_model=SolicitacaoCreate)
async def criar_solicitacao(poles_request: PolesRequest, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_session), usuario_logado: UsuarioModel = Depends(get_current_user)):
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

        return {"id": nova_solicitacao.id, "status": nova_solicitacao.status, "postes": nova_solicitacao.postes, "imagens": nova_solicitacao.imagens}
