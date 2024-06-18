from typing import Optional, List
from pydantic import BaseModel, EmailStr
from schemas.solicitacao_schema import SolicitacaoSchema



class UsuarioSchemaBase(BaseModel):
    id: Optional[int] = None
    email: EmailStr
    admin: bool = False

    class Config:
        from_attributes = True


class UsuarioSchemaCreate(UsuarioSchemaBase):
    senha: str


class UsuarioSchemaSolicitacoes(UsuarioSchemaBase):
    solicitacoes: Optional[List[SolicitacaoSchema]]


class UsuarioSchemaUp(UsuarioSchemaBase):
    email: Optional[EmailStr]
    senha: Optional[str]
    admin: Optional[bool]