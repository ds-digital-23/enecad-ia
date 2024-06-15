from typing import Optional
from pydantic import BaseModel
from datetime import datetime



class ModeloSchema(BaseModel):
    id: Optional[int] = None
    modelo_id: int
    nome: str
    descricao: str
    modelo_nome: str
    status: int
    criado_em: datetime
    atualizado_em: datetime

    class Config:
        from_attributes = True


class ModeloCreate(BaseModel):
    modelo_id: int
    nome: str
    descricao: str
    modelo_nome: str
    status: int


class ModeloUpdate(BaseModel):
    modelo_id: Optional[int] = None
    nome: Optional[str] = None
    descricao: Optional[str] = None
    modelo_nome: Optional[str] = None
    status: Optional[int] = None
    usuario_id: Optional[int] = None

    class Config:
        from_attributes = True


class ModeloResponse(BaseModel):
    nome: str
    modelo_nome: str

    class Config:
        from_attributes = True