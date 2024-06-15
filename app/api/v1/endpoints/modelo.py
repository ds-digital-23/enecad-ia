from typing import List
from fastapi import APIRouter, status, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from models.modelo_model import ModeloModel
from models.usuario_model import UsuarioModel
from schemas.modelo_schema import ModeloSchema, ModeloCreate, ModeloUpdate
from core.deps import get_session, get_current_user



router = APIRouter()


@router.post('/', status_code=status.HTTP_201_CREATED, response_model=ModeloCreate)
async def post_modelo(modelo: ModeloCreate, db: AsyncSession = Depends(get_session), usuario_logado: UsuarioModel = Depends(get_current_user)):
    novo_modelo: ModeloModel = ModeloModel(
        modelo_id=modelo.modelo_id, 
        nome= modelo.nome,
        descricao=modelo.descricao, 
        modelo_nome=modelo.modelo_nome,
        status=modelo.status, 
        usuario_id=usuario_logado.id
    )
    db.add(novo_modelo)
    await db.commit()

    return novo_modelo


@router.get('/', response_model=List[ModeloSchema])
async def get_modelos(db: AsyncSession = Depends(get_session), usuario_logado: UsuarioModel = Depends(get_current_user)):
    async with db as session:
        query = select(ModeloModel)
        result = await session.execute(query)
        modelos: List[ModeloModel] = result.scalars().unique().all()

        return modelos
    

@router.get('/{id}', response_model=ModeloSchema, status_code=status.HTTP_200_OK)
async def get_modelo(id: int, db: AsyncSession = Depends(get_session), usuario_logado: UsuarioModel = Depends(get_current_user)):
    async with db as session:
        query = select(ModeloModel).filter(ModeloModel.id == id)
        result = await session.execute(query)
        modelo = result.scalars().unique().one_or_none()

        if modelo:
            return modelo
        else:
            raise HTTPException(detail='Modelo não encontrado.', status_code=status.HTTP_404_NOT_FOUND)
        

@router.put('/{id}', response_model=ModeloUpdate, status_code=status.HTTP_202_ACCEPTED)
async def put_modelo(id: int, modelo: ModeloUpdate, db: AsyncSession = Depends(get_session), usuario_logado: UsuarioModel = Depends(get_current_user)):
    async with db as session:
        query = select(ModeloModel).filter(ModeloModel.id == id)
        result = await session.execute(query)
        modelo_up = result.scalars().unique().one_or_none()
    
        if modelo_up:
            update_data = modelo.model_dump(exclude_unset=True)
            for key, value in update_data.items():
                setattr(modelo_up, key, value)
            await session.commit()
            await session.refresh(modelo_up)
            return modelo_up
        else:
            raise HTTPException(detail='Modelo não encontrado.', status_code=status.HTTP_404_NOT_FOUND)
        