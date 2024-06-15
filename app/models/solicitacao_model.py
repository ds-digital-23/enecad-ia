from core.configs import settings
from sqlalchemy import Column, Integer, String, DateTime, func, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import List


class SolicitacaoModel(settings.DBBaseModel):
    __tablename__ = 'solicitacoes'

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    status: str = Column(String(50), nullable=False)
    postes: int = Column(Integer, nullable=False, default=0)
    imagens: int = Column(Integer, nullable=False, default=0)
    criado_em: datetime = Column(DateTime, server_default=func.now(), nullable=False)
    atualizado_em: datetime = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    
    usuario_id: int = Column(Integer, ForeignKey('usuarios.id'))
    criador = relationship('UsuarioModel', back_populates='solicitacoes', lazy='joined')
    