from core.configs import settings
from sqlalchemy import Column, Integer, DateTime, func, JSON
from datetime import datetime



class ArquivoModel(settings.DBBaseModel):
    __tablename__ = 'arquivos'

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    solicitacao_id: int = Column(Integer, nullable=False)
    json: str = Column(JSON, nullable=False)
    atualizado_em: datetime = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    
    