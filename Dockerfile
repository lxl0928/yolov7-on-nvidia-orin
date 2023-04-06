FROM nvcr.io/nvidia/l4t-ml:r34.1.1-py3

RUN mkdir -p /app
WORKDIR /app
copy . /app
RUN rm -rf .git/

RUN pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple/
RUN pip3 install -r requirements.txt

EXPOSE 8012
CMD ["python3", "api.py"]