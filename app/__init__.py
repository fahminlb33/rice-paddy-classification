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
model_path = os.path.abspath(os.path.join(base_directory, "model"))
temp_path = os.path.abspath(os.path.join(base_directory, "temp"))

# create predictor instance
predictor_model = PredictorModel()

print("Loading model from:", model_path)
predictor_model.load_model(model_path, MODEL_FILENAME, CLASS_FILENAME)

# init fastapi
app = FastAPI()

# init templates
templates = Jinja2Templates(directory=os.path.join(base_directory, "templates"))

# init static files
app.mount("/assets", StaticFiles(directory=os.path.join(base_directory, "assets")), name="assets")
app.mount("/temp", StaticFiles(directory=os.path.join(base_directory, "temp")), name="temp")

# --- routes ---

@app.get("/", response_class=HTMLResponse)
async def index_route(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/prediction", response_class=HTMLResponse)
async def prediction_route(request: Request, file: UploadFile):
    # delete old files
    for file in os.listdir(temp_path):
      os.remove(os.path.join(temp_path, file))

    # save file to temp directory
    uploaded_path = os.path.abspath(os.path.join(temp_path, file.filename))
    async with aiofiles.open(uploaded_path, 'wb') as temp_stream:
        content = await file.read()
        await temp_stream.write(content)

    # make prediction
    heatmap_path = os.path.abspath(os.path.join(temp_path, "gradcam-" + file.filename))
    (prediction, heatmap_path) = predictor_model.predict(uploaded_path, heatmap_path)

    return templates.TemplateResponse("prediction.html", {
      "request": request,
      "predicted": prediction,
      "original_image": "uploaded.png",
      "filtered_image": "output.png"
    })
