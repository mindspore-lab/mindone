import asyncio
import concurrent.futures
import logging
import os
import random
import tempfile
import traceback
import uuid
from typing import Optional

import aiohttp
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import mindspore as ms

from mindone.diffusers.pipelines.stable_diffusion_3 import StableDiffusion3Pipeline

logger = logging.getLogger(__name__)


class TextToImageInput(BaseModel):
    model: str
    prompt: str
    size: Optional[str] = None
    n: Optional[int] = None


class HttpClient:
    session: aiohttp.ClientSession = None

    def start(self):
        self.session = aiohttp.ClientSession()

    async def stop(self):
        await self.session.close()
        self.session = None

    def __call__(self) -> aiohttp.ClientSession:
        assert self.session is not None
        return self.session


class TextToImagePipeline:
    pipeline: StableDiffusion3Pipeline = None

    def start(self):
        if ms.device_context.ascend.is_available():
            model_path = os.getenv("MODEL_PATH", "stabilityai/stable-diffusion-3.5-large")
            logger.info("Loading Ascend")
            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_path,
                mindspore_dtype=ms.bfloat16,
            )
        else:
            raise Exception("No Ascend device available")


app = FastAPI()
service_url = os.getenv("SERVICE_URL", "http://localhost:8000")
image_dir = os.path.join(tempfile.gettempdir(), "images")
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
app.mount("/images", StaticFiles(directory=image_dir), name="images")
http_client = HttpClient()
shared_pipeline = TextToImagePipeline()
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# Configure CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods, e.g., GET, POST, OPTIONS, etc.
    allow_headers=["*"],  # Allows all headers
)


@app.on_event("startup")
def startup():
    http_client.start()
    shared_pipeline.start()


def save_image(image):
    filename = "draw" + str(uuid.uuid4()).split("-")[0] + ".png"
    image_path = os.path.join(image_dir, filename)
    # write image to disk at image_path
    logger.info(f"Saving image to {image_path}")
    image.save(image_path)
    return os.path.join(service_url, "images", filename)


@app.get("/")
@app.post("/")
@app.options("/")
async def base():
    return "Welcome to MindONE-Diffusers! Where you can use diffusion models to generate images"


@app.post("/v1/images/generations")
async def generate_image(image_input: TextToImageInput):
    try:
        loop = asyncio.get_event_loop()
        scheduler = shared_pipeline.pipeline.scheduler.from_config(shared_pipeline.pipeline.scheduler.config)
        pipeline = StableDiffusion3Pipeline.from_pipe(shared_pipeline.pipeline, scheduler=scheduler)
        generator = np.random.Generator(np.random.PCG64(random.randint(0, 10000000)))
        output = await loop.run_in_executor(thread_pool, lambda: pipeline(image_input.prompt, generator=generator))
        logger.info(f"output: {output}")
        image_url = save_image(output[0][0])
        return {"data": [{"url": image_url}]}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        elif hasattr(e, "message"):
            raise HTTPException(status_code=500, detail=e.message + traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e) + traceback.format_exc())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
