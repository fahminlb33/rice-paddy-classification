import os
import logging
from logging.config import dictConfig


from app.common_helpers import logging_config

# initialize logger
logger = logging.getLogger()
dictConfig(logging_config)

import aiofiles
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.predictor import PredictorService
from app.common_helpers import is_file_allowed

# get model name
MODEL_FILENAME = os.environ.get("MODEL_NAME", "tensorflow.h5")
CLASS_FILENAME = os.environ.get("CLASS_NAME", "class_names.z")

logger.info("Environment variables:")
logger.info(f"MODEL_FILENAME: {MODEL_FILENAME}")
logger.info(f"CLASS_FILENAME: {CLASS_FILENAME}")

# get base directory relative to this file
base_directory = os.path.dirname(os.path.abspath(__file__))
temp_path = os.path.abspath(os.path.join(base_directory, "temp"))
model_path = os.path.abspath(os.path.join(base_directory, "model"))
assets_path = os.path.abspath(os.path.join(base_directory, "assets"))
template_path = os.path.abspath(os.path.join(base_directory, "templates"))

# create predictor instance
predictor_model = PredictorService()

logger.info(f"Loading model from: {model_path}")
predictor_model.load_model(model_path, MODEL_FILENAME, CLASS_FILENAME)

# init fastapi
app = FastAPI()
app.logger = logger

# init templates
templates = Jinja2Templates(directory=template_path)

# init static files
app.mount("/temp", StaticFiles(directory=temp_path), name="temp")
app.mount("/assets", StaticFiles(directory=assets_path), name="assets")


def get_css_for_prediction(prediction: str) -> str:
    if "HEALTHY" in prediction.upper():
        return "has-background-success-dark has-text-success-light"
    else:
        return "has-background-danger-dark has-text-danger-light"

# --- routes ---

@app.get("/", response_class=HTMLResponse)
async def index_route(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/prediction", response_class=HTMLResponse)
async def prediction_route(request: Request, file: UploadFile):
    # create temporary directory
    _, file_extension = os.path.splitext(file.filename)
    uploaded_path = os.path.join(temp_path, "uploaded" + file_extension)

    # check file extension
    if not is_file_allowed(file_extension):
        logger.warning(f"File extension not allowed: {file_extension}")
        return templates.TemplateResponse("error.html",
            {
                "request": request,
                "error_message": "File extension not allowed",
                "css_class": "has-background-danger-dark has-text-danger-light",
            },
        )

    # save file to temp directory
    async with aiofiles.open(uploaded_path, 'wb') as temp_stream:
        content = await file.read()
        await temp_stream.write(content)

    # get file size
    file_size = os.path.getsize(uploaded_path)
    logger.info(f"Uploaded file size: {file_size}")

    # resize image to maximum of MAX_HEIGHT
    logger.info(f"Constraining uploaded image size...")
    resized_path = predictor_model.constrain_image_size(uploaded_path)

    # make prediction
    logger.info(f"Running prediction...")
    (prediction, heatmap_path, superimposed_path, masked_path) \
        = predictor_model.predict(resized_path, temp_path)

    return templates.TemplateResponse("prediction.html", {
        "request": request,
        "predicted": prediction,
        "background_css": get_css_for_prediction(prediction),
        "original_image": os.path.basename(resized_path),
        "heatmap_image": os.path.basename(heatmap_path),
        "superimposed_image": os.path.basename(superimposed_path),
        "masked_image": os.path.basename(masked_path)
    })
