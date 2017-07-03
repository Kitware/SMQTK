FROM node:8
MAINTAINER omar.padron@kitware.com

RUN mkdir -p /app
COPY package.json content-type-map.js index.js /app/
RUN cd /app && npm install

VOLUME /data /newdata /links
ENTRYPOINT ["node", "/app/index.js"]

EXPOSE 12345
