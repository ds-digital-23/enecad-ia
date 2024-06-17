from typing import List, Dict, Optional
from pydantic import BaseModel, HttpUrl


class PhotoSchema(BaseModel):
    PhotoId: int
    URL: str


class PoleSchema(BaseModel):
    PoleId: int
    Photos: List[PhotoSchema]

    class Config:
        from_attributes = True


class PolesRequest(BaseModel):
    Poles: List[PoleSchema]
    webhook_url: Optional[HttpUrl] = None

    class Config:
        from_attributes = True


class Resultado(BaseModel):
    PhotoId: int
    URL: str
    Resultado: Dict[str, bool]


class SolicitacaoSchema(BaseModel):
    id: int
    status: str
    poles: List[PoleSchema]

    class Config:
        from_attributes = True


class SolicitacaoCreate(BaseModel):
    id: int
    status: str
    postes: int
    imagens: int
