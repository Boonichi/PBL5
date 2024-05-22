import io
from PIL import Image
from typing import Annotated
import uvicorn
import requests
import argparse
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from model_predictor import ModelPredictor
from pydantic import BaseModel
from typing import List
from configs import get_args_parser
import os


class ReturnedObject(BaseModel):
    medical_leaf_name: str


IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))

# Đường dẫn tới thư mục public
public_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "public")


class AppAPI:
    def __init__(self, args: dict) -> None:
        self.app = FastAPI()
        self.app.mount(
            "/dataset/fold_0", StaticFiles(directory=IMAGE_DIR), name="images"
        )
        self.predictor = ModelPredictor(args)

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.get("/")
        async def root():
            return {"message": "hello"}

        @self.app.post("/classify", response_model=ReturnedObject)
        async def classify(image_file: UploadFile = File(...)):
            print(image_file)
            request_object_content = await image_file.read()
            image = Image.open(io.BytesIO(request_object_content))
            output = self.predictor.predict(image)
            return output

    def run(self, port: int):
        uvicorn.run(self.app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    app = AppAPI(args)

    app.run(port=8000)
