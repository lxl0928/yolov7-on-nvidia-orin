# FROM nvcr.io/nvidia/l4t-ml:r34.1.1-py3
# FROM yolov7-on-nvidia-orin_web:latest
FROM yolov7_arm64:v0.0.1

RUN mkdir -p /app
WORKDIR /app
copy . /app

# OPENCV BUG
copy cv_init.py /usr/local/lib/python3.8/dist-packages/cv2/__init__.py
RUN rm -rf .git/

RUN apt-get update && apt-get install vim -y

RUN pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple/
RUN pip3 install -r requirements.txt
RUN /app/cv_init.py /usr/local/lib/python3.8/dist-packages/cv2/__init__.py   # OPENCV BUG

EXPOSE 8012
CMD ["top"]
# CMD ["python3", "api.py"]
