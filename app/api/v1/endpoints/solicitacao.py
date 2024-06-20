import logging
import asyncio
import time
import os
import json
import gdown
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

router = APIRouter()


# Carregar o modelo YOLO no início
#MODEL_PATH = 'app/ia/model_ip_v1.3.pt'
#model = YOLO(MODEL_PATH)

from main import model

async def predict_image(image_url: str) -> Dict[str, float]:
    result = await asyncio.to_thread(model.predict, image_url, stream=True)
    max_conf = round(max((res.conf.item() for res in result.boxes), default=0), 3)
    return {"url": image_url, "max_conf": max_conf}

async def detect_images(solicitacao_id: int, session: AsyncSession, poles_request: PolesRequest):
    async with session:
        
        # Coletar todas as URLs de imagens
        image_urls = []
        for pole in poles_request.Poles:
            for photo in pole.Photos:
                image_urls.append(photo.URL)

        # Predizer para todas as imagens
        predictions = await asyncio.gather(*(predict_image(url) for url in image_urls))

        # Agrupar resultados por poste
        resultado = {"Poles": []}
        for pole in poles_request.Poles:
            pole_result = {"PoleId": pole.PoleId, "Photos": []}
            for photo in pole.Photos:
                for prediction in predictions:
                    if prediction["url"] == photo.URL:
                        pole_result["Photos"].append({
                            "PhotoId": photo.PhotoId,
                            "URL": photo.URL,
                            "max_conf": prediction["max_conf"]
                        })
            resultado["Poles"].append(pole_result)

        response_file_path = os.path.join('results', f"solicitacao_{solicitacao_id}.json")
        with open(response_file_path, 'w') as file:
            json.dump(resultado, file)


async def start_detection(solicitacao_id: int, session: AsyncSession, poles_request: PolesRequest):
    try:
        detection_results = await detect_images(solicitacao_id, session, poles_request)
        await update_status(solicitacao_id=solicitacao_id, status='Concluído', db=session)
        return detection_results
    except Exception as e:
        logging.error(f"Detection task failed: {e}")
        await update_status(solicitacao_id=solicitacao_id, status='Falhou', db=session)

@router.post("/", response_model=SolicitacaoCreate)
async def criar_solicitacao(poles_request: PolesRequest, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_session), usuario_logado: UsuarioModel = Depends(get_current_user)):
    start_time = time.time()
    total_poles = len(poles_request.Poles)
    total_photos = sum(len(pole.Photos) for pole in poles_request.Poles)

    if total_poles > 1000:
        raise HTTPException(status_code=400, detail="Número de postes não pode ser maior que 1000.")

    nova_solicitacao = SolicitacaoModel(status="Em andamento", postes=total_poles, imagens=total_photos, usuario_id=usuario_logado.id)

    async with db as session:
        session.add(nova_solicitacao)
        await session.commit()
        await session.refresh(nova_solicitacao)

        background_tasks.add_task(start_detection, nova_solicitacao.id, session, poles_request)

    end_time = time.time()
    logging.info(f"Processed request in {end_time - start_time:.2f} seconds")
    return {"id": nova_solicitacao.id, "status": nova_solicitacao.status, "postes": nova_solicitacao.postes, "imagens": nova_solicitacao.imagens}

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
