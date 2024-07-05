# -*- coding: utf-8 -*-
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from core.configs import settings
from api.v1.api import api_router
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(
    title='Enecad - API',
    version='1.0.0',
    description='API desenvolvida para a Enecad a fim de detectar objetos em rede de distribuição via modelos de visão computacional.',
    license="Licença Comercial Enecad"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ajuste conforme necessário
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos os métodos (POST, GET, etc.)
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    content = {"mensagem": "Bem-vindo à Enecad-API para detecção de objetos em rede de distribuição!"}
    return JSONResponse(content=content, media_type="application/json; charset=utf-8")


app.include_router(api_router, prefix=settings.API_VERSION)
