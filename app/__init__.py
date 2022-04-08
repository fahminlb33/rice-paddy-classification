import os

import aiofiles
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.predictor import PredictorModel

MODEL_FILENAME = os.environ.get("MODEL_NAME", "tensorflow.h5")
CLASS_FILENAME = os.environ.get("CLASS_NAME", "class_names.z")

# get base directory relative to this file
base_directory = os.path.dirname(os.path.abspath(__file__))
temp_path = os.path.abspath(os.path.join(base_directory, "temp"))
model_path = os.path.abspath(os.path.join(base_directory, "model"))
assets_path = os.path.abspath(os.path.join(base_directory, "assets"))
template_path = os.path.abspath(os.path.join(base_directory, "templates"))

# create predictor instance
predictor_model = PredictorModel()

print("Loading model from:", model_path)
predictor_model.load_model(model_path, MODEL_FILENAME, CLASS_FILENAME)

# init fastapi
app = FastAPI()

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
    # save file to temp directory
    _, file_extension = os.path.splitext(file.filename)
    uploaded_path = os.path.join(temp_path, "uploaded" + file_extension)

    async with aiofiles.open(uploaded_path, 'wb') as temp_stream:
        content = await file.read()
        await temp_stream.write(content)

    # make prediction
    (prediction, heatmap_path, heatmap_imposed_path, masked_img) = predictor_model.predict(uploaded_path, temp_path)

    return templates.TemplateResponse("prediction.html", {
        "request": request,
        "predicted": prediction,
        "background_css": get_css_for_prediction(prediction),
        "original_image": os.path.basename(uploaded_path),
        "heatmap_image": os.path.basename(heatmap_path),
        "heatmap_imposed_image": os.path.basename(heatmap_imposed_path),
        "masked_image": os.path.basename(masked_img)
    })
