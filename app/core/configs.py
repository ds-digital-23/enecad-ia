from typing import ClassVar
from pydantic_settings import BaseSettings
from sqlalchemy.ext.declarative import declarative_base
from decouple import config


class Settings(BaseSettings):
    API_VERSION: str = config('API_VERSION')
    TEST_MODE: bool = config('TEST_MODE', default=False, cast=bool)
    DB_URL: str = config('DB_URL_TEST') if TEST_MODE else config('DB_URL')
    DBBaseModel: ClassVar = declarative_base()

    JWT_SECRET: str = config('JWT_SECRET')
    ALGORITHM: str = config('ALGORITHM')
    ACCESS_TOKEN_EXPIRE_MINUTES : int = 60 * 24 * 7 * 52

    class Config:
        case_sensitive = True


settings: Settings = Settings()