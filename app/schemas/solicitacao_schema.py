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
    Models: Optional[List[str]] = ["Poste", "UM", "BT", "IP", "MT", "Equipamentos"]
    webhook_url: Optional[HttpUrl] = None

    class Config:
        from_attributes = True

class DetectionResultSingleClass(BaseModel):
    detected: bool
    max_confidence: float

class DetectionResultMultiClass(BaseModel):
    classes: Dict[str, float]

class DetectionResult(BaseModel):
    single_class: Optional[DetectionResultSingleClass] = None
    multi_class: Optional[DetectionResultMultiClass] = None

class Resultado(BaseModel):
    PhotoId: int
    URL: str
    Resultado: Dict[str, DetectionResult]

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
