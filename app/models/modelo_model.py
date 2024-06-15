from core.configs import settings
from sqlalchemy import Column, Integer, String, DateTime, func, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime


class ModeloModel(settings.DBBaseModel):
    __tablename__ = 'modelos'

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    modelo_id: int = Column(Integer, nullable=False)
    nome: str = Column(String(256), nullable=False)
    descricao: str = Column(String(256), nullable=False)
    modelo_nome: str = Column(String(256), nullable=False)
    status: int = Column(Integer, nullable=False)
    criado_em: datetime = Column(DateTime, server_default=func.now(), nullable=False)
    atualizado_em: datetime = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    
    usuario_id: int = Column(Integer, ForeignKey('usuarios.id'))
    criador = relationship('UsuarioModel', back_populates='modelos', lazy='joined')
