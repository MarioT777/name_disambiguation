FROM python:3.8

# 作者
MAINTAINER MarioT

WORKDIR ./docker_demo

ADD . .

RUN pip install -r requirements.txt

CMD ["python", "./src/main.py"]