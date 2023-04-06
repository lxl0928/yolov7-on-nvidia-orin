# coding: utf-8

import os
import json
import uvicorn
from typing import List, Optional
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel, Field

from detect import common_detect

app = FastAPI()


class RequestBody(BaseModel):
    imgDir: Optional[str] = Field(description="测试图片在服务器上的路径")
    imgUrls: Optional[List[str]] = Field(description="测试图片网络路径")


@app.post("/yolov7/test")
async def detect(request_body: RequestBody):
    result_dic = {}

    if request_body.imgDir:
        print(request_body.imgDir)
        if os.path.exists(request_body.imgDir):
            print("imgDir exists")
            result_dic = common_detect(source=request_body.imgDir, no_trace=True)
            print(result_dic)
    return Response(status_code=200, content=json.dumps(result_dic))


if __name__ == '__main__':

    uvicorn.run(
        app="api:app",
        host="0.0.0.0",
        port=8012,
        limit_concurrency=100,
        workers=8,
        log_level="info",
        reload=True
    )
