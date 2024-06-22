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



async def predict_model(model, images):
    return await asyncio.to_thread(model.predict, images)


async def process_pole(pole) -> Dict:
    images = [photo.URL for photo in pole.Photos]
    photo_ids = [photo.PhotoId for photo in pole.Photos]
    
    # Realiza predição de forma assíncrona e paralela para os dois modelos
    results_ip, results_la = await asyncio.gather(
        predict_model(loaded_models["model_ip_v1.3.pt"]["model"], images),
        predict_model(loaded_models["model_la_v1.2.pt"]["model"], images)
    )
    
    # Processa os resultados
    pole_results = []
    for photo_id, image, result_ip, result_la in zip(photo_ids, images, results_ip, results_la):
        max_conf_ip = round(max((box.conf.item() for res in result_ip for box in res.boxes), default=0), 3)
        max_conf_la = round(max((box.conf.item() for res in result_la for box in res.boxes), default=0), 3)

        pole_result = {
            loaded_models["model_ip_v1.3.pt"]["nome"]: {
                "detected": max_conf_ip > 0,
                "max_confidence": max_conf_ip
            },
            loaded_models["model_la_v1.2.pt"]["nome"]: {
                "detected": max_conf_la > 0,
                "max_confidence": max_conf_la
            }
        }
        pole_results.append(Resultado(PhotoId=photo_id, URL=image, Resultado=pole_result))
    
    output = {"PoleId": pole.PoleId, "Photos": [result.model_dump() for result in pole_results]}
    gc.collect()  
    return output




async def detect_objects(request: PolesRequest, solicitacao_id: int): 
    start_time = time.time()
    pole_tasks = [process_pole(pole) for pole in request.Poles]
    pole_results = await asyncio.gather(*pole_tasks)
    print(pole_results)
    
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
    gc.collect() 
    return response


async def start_detection(solicitacao_id: int, db: AsyncSession, poles_request: PolesRequest):
    start_time = time.time()
    global modelos
    modelos = list(loaded_models.keys())
    print('Modelos carregados em solicitacao', modelos)

    async with db as session:
        try: 

            max_retries = 2
            for attempt in range(max_retries):
                try:
                    detection_results = await detect_objects(request=poles_request, solicitacao_id=solicitacao_id)
                except AttributeError as e:
                    print(f"Erro ao executar detection results: {e}")
                    if attempt < max_retries - 1:
                        print("Tentando novamente...")
                        await asyncio.sleep(1)
                    else:
                        raise e

            await update_status(solicitacao_id=solicitacao_id, status='Concluído', db=session)
            end_time = time.time()
            print(f"start_detection function took {end_time - start_time:.2f} seconds")
            gc.collect() 
            return detection_results
        
        except Exception as e:
            await update_status(solicitacao_id=solicitacao_id, status='Falhou', db=session)
            end_time = time.time()
            print(f"start_detection function took {end_time - start_time:.2f} seconds")
            gc.collect() 
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
