from fastapi import APIRouter
from api.v1.endpoints import usuario, solicitacao, modelo



api_router = APIRouter()
api_router.include_router(usuario.router, prefix='/usuarios', tags=['Usuários'])
api_router.include_router(solicitacao.router, prefix='/solicitacoes', tags=['Solicitações'])
api_router.include_router(modelo.router, prefix='/modelos', tags=['Modelos'])
