FROM kitware/smqtk
MAINTAINER omar.padron@kitware.com

# Add apt repo for postgres
RUN apt-key adv --keyserver ha.pool.sks-keyservers.net --recv-keys \
    B97B0AFCAA1A47F044F244A07FCC7D46ACCC4CF8 && \
    echo "deb http://apt.postgresql.org/pub/repos/apt/ trusty-pgdg main" >> \
    /etc/apt/sources.list && \
    apt-get update && apt-get install -y -q postgresql-client

COPY custom-entry-point.sh /app/scripts

ENTRYPOINT ["/app/scripts/custom-entry-point.sh"]

