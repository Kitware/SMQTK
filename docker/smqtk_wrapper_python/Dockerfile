FROM python:3
MAINTAINER omar.padron@kitware.com

RUN mkdir -p /app
COPY requirements.txt main.py /app/
RUN cd /app && pip install -r requirements.txt

VOLUME /data /newdata /links
ENTRYPOINT ["python", "/app/main.py"]

EXPOSE 12345
