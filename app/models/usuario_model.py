from core.configs import settings
from sqlalchemy import Column, Integer, String, Boolean
from sqlalchemy.orm import relationship


class UsuarioModel(settings.DBBaseModel):
    __tablename__ = 'usuarios'

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    email: str = Column(String(256), index=True, nullable=False, unique=True)
    senha: str = Column(String(256), nullable=False)
    admin: bool = Column(Boolean, default=False)
    
    modelos = relationship('ModeloModel', cascade='all, delete-orphan', back_populates='criador', uselist=True, lazy='joined')
    solicitacoes = relationship('SolicitacaoModel', cascade='all, delete-orphan', back_populates='criador', uselist=True, lazy='joined')
    