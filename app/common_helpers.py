from logging.config import dictConfig

# create formatter
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default_formatter": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "formatter": "default_formatter",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout"
        }
    },
    "loggers": {
        "": {
            "handlers": ["console"],
            "level": "DEBUG"
        }
    },
}

def init_logger():
    dictConfig(logging_config)

def is_file_allowed(file_extension: str) -> bool:
    return file_extension.lower() in [".jpg", ".jpeg", ".jfif", ".pjpeg", ".pjp", ".png"]
