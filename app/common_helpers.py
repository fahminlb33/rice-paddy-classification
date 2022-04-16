import os
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
        },
        "azure": {
            "level": "DEBUG",
            "formatter": "default",
            "class": "opencensus.ext.azure.log_exporter.AzureLogHandler",
            "connection_string": os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING", "InstrumentationKey=00000000-0000-0000-0000-000000000000")
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
            "handlers": ["default", "azure"],
            "propagate": False
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["default", "azure"]
    }
}

def is_file_allowed(file_extension: str) -> bool:
    return file_extension.lower() in [".jpg", ".jpeg", ".jfif", ".pjpeg", ".pjp", ".png"]
