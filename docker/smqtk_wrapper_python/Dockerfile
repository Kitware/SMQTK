FROM python:3
MAINTAINER omar.padron@kitware.com

RUN mkdir -p /app
ADD requirements.txt /app/
RUN cd /app && pip install -r requirements.txt
ADD content_type_map.py /app/
ADD main.py /app/
VOLUME /data /newdata /links
ENTRYPOINT ["python", "/app/main.py"]

EXPOSE 12345
EXPOSE 12346
