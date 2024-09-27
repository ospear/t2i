from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi import FastAPI, Form

from t2i.controller import TextToImageController
from t2i.logger import setup_logger
from t2i.text_to_image_item import TextToImageItem, SDXL1

setup_logger()
app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/text-to-images")
async def create_image(
    prompt: str = Form(),
    negative_prompt: str = Form(),
    num_inference_steps: int = Form(default=30),
    guidance_scale: int = Form(default=7.0),
    model: str = Form(default=SDXL1),
):
    item = TextToImageItem(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        model=model,
    )
    path = TextToImageController().create_image(item)
    return FileResponse(path=path)
