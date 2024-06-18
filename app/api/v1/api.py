from fastapi import APIRouter
from api.v1.endpoints import usuario
from api.v1.endpoints import solicitacao



api_router = APIRouter()
api_router.include_router(usuario.router, prefix='/usuarios', tags=['Usuários'])
api_router.include_router(solicitacao.router, prefix='/solicitacoes', tags=['Solicitações'])
