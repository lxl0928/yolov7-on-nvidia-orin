# coding: utf-8

import os
import json
import uvicorn
from typing import List, Optional
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel, Field

from detect import common_detect


import torch
from models.experimental import attempt_load
from utils.torch_utils import TracedModel, select_device

app = FastAPI()
device = select_device('0')  # or 'cpu'
common_detect_model = attempt_load("./yolov7.pt", map_location=device)  # load FP32 model
trace, half, img_size = False, False, 640
if trace:
    common_detect_model = TracedModel(common_detect_model, device, img_size)

if half:
    common_detect_model.half()  # to FP16


class RequestBody(BaseModel):
    imgDir: Optional[str] = Field(description="测试图片在服务器上的路径")
    imgUrls: Optional[List[str]] = Field(description="测试图片网络路径")


@app.get("/")
async def root(request: Request):
    return Response(status_code=200, content=json.dumps({"message": "Hello World"}))


@app.post("/yolov7/test")
async def detect(request_body: RequestBody):
    if request_body.imgDir:
        print(request_body.imgDir)
        if os.path.exists(request_body.imgDir):
            print("imgDir exists")
            result_dic = common_detect(source=request_body.imgDir, common_detect_model=common_detect_model)
            print(result_dic)

    if request_body.imgUrls:
        print(request_body.imgUrls)

    return Response(status_code=200, content=json.dumps(result_dic))


if __name__ == '__main__':
    uvicorn.run(app="api:app", host="0.0.0.0", port=8012)
