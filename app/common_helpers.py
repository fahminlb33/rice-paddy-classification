import sys

# create formatter
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelname)s:\t%(name)s - %(message)s"
        }
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": sys.stderr
        },
        "access": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": sys.stdout
        }
    },
    "loggers": {
        "uvicorn.error": {
            "level": "INFO",
            "handlers": ["default"],
            "propagate": False
        },
        "uvicorn.access": {
            "level": "INFO",
            "handlers": ["default"],
            "propagate": False
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["default"]
    }
}

def is_file_allowed(file_extension: str) -> bool:
    return file_extension.lower() in [".jpg", ".jpeg", ".jfif", ".pjpeg", ".pjp", ".png"]
