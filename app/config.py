from pydantic import BaseSettings

class Settings(BaseSettings):
    model_name: str = "tensorflow.h5"
    class_name: str = "class_names.z"

settings = Settings()
