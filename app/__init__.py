import os
import logging
from logging.config import dictConfig
from typing import List, Tuple

from app.common_helpers import logging_config

# initialize logger
dictConfig(logging_config)
logger = logging.getLogger(__name__)

import aiofiles
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from app.predictor import PredictorService
from app.common_helpers import is_file_allowed
from app.config import settings

# get base directory relative to this file
base_directory = os.path.dirname(os.path.abspath(__file__))
temp_path = os.path.abspath(os.path.join(base_directory, "temp"))
model_path = os.path.abspath(os.path.join(base_directory, "model"))
assets_path = os.path.abspath(os.path.join(base_directory, "assets"))
template_path = os.path.abspath(os.path.join(base_directory, "templates"))

# create predictor instance
predictor_model = PredictorService()

logger.info(f"Loading model from: {model_path}")
predictor_model.load_model(model_path, settings.model_name, settings.class_name)

# init fastapi
app = FastAPI()
app.logger = logger

# add CORS middleware
origins = [
    "https://kodesiana.com",
    "https://www.kodesiana.com",
    "https://skripsi.kodesiana.com",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# init templates
templates = Jinja2Templates(directory=template_path)

# init static files
app.mount("/temp", StaticFiles(directory=temp_path), name="temp")
app.mount("/assets", StaticFiles(directory=assets_path), name="assets")


def get_css_for_prediction(prediction: str) -> str:
    if prediction == "HEALTHY":
        return "has-background-success-dark has-text-success-light"
    else:
        return "has-background-danger-dark has-text-danger-light"

def get_probability_tuples(predictions: List[float]) -> List[Tuple[str, float]]:
    # map prediction to class name
    class_proba_dict = {predictor_model.get_class_from_prediction(i): v for i, v in enumerate(predictions)}

    # sort by probability
    class_proba_dict = {k: v for k, v in sorted(class_proba_dict.items(), key=lambda item: item[1], reverse=True)}

    # convert to list of tuples
    for i, (pred_class, proba) in enumerate(class_proba_dict.items()):
        if i == 0 and pred_class == "HEALTHY":
            yield ("is-success", pred_class, f"{proba:.2f}%")
        elif i == 0 and pred_class != "HEALTHY":
            yield ("is-danger", pred_class, f"{proba:.2f}%")
        else:
            yield ("", pred_class, f"{proba:.2f}%")

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
    (prediction_proba, heatmap_path, superimposed_path, masked_path) \
        = predictor_model.predict(resized_path, temp_path)

    # collapse prediction results
    predicted_class = predictor_model.get_most_likely_class(prediction_proba)
    class_proba_tuples = get_probability_tuples(prediction_proba.tolist())

    return templates.TemplateResponse("prediction.html", {
        "request": request,
        "predicted": predicted_class,
        "probabilities": class_proba_tuples,
        "background_css": get_css_for_prediction(predicted_class),
        "original_image": os.path.basename(resized_path),
        "heatmap_image": os.path.basename(heatmap_path),
        "superimposed_image": os.path.basename(superimposed_path),
        "masked_image": os.path.basename(masked_path)
    })
