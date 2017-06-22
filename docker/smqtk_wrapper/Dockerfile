FROM node:8
MAINTAINER omar.padron@kitware.com

RUN mkdir -p /app
ADD package.json /app/
RUN cd /app && npm install
ADD content-type-map.js /app/
ADD index.js /app/
VOLUME /data /newdata /links
ENTRYPOINT ["node", "/app/index.js"]

EXPOSE 12345
EXPOSE 12346
