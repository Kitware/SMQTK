FROM ubuntu:14.04
MAINTAINER Aashish Chaudhary <aashish.chaudhary@kitware.com>

# The environment variables beginning with KWDEMO can be used to map this demo
# to the main url space
ENV KWDEMO_READY FALSE
ENV KWDEMO_NAME GeoinfoSpace Demo
ENV KWDEMO_KEY geoinfoSpace
ENV KWDEMO_SRCURL https://github.com/memex-explorer/geoinfo_space
ENV KWDEMO_DESC Memex geoinfo data explorer
ENV KWDEMO_IMG "geoinfospace.png"

EXPOSE 8001

RUN apt-get update && apt-get install -y curl \
    software-properties-common \
    python-software-properties \
    build-essential \
    libffi-dev \
    libpython-dev \
    python-pip \
    postgresql-client \
    postgresql-common \
    libpq-dev \
    vim \
    git

# Get nodejs apt source
RUN curl -sL https://deb.nodesource.com/setup | sudo bash -
# Add mongo apt source
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 7F0CEB10 && \
    echo "deb http://repo.mongodb.org/apt/ubuntu "$(lsb_release -sc)"/mongodb-org/3.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-3.0.list
# Install packages requiring additional sources
RUN apt-get update && apt-get install -y \
    mongodb-org-shell \
    mongodb-org-tools \
    nodejs

RUN pip install girder python-dateutil PyGreSQL==4.1.1 psycopg2 && \
    python -c "import bcrypt" && \
    girder-install web && \
    girder-install plugin

RUN pip install geopy

RUN npm install -g grunt-cli

# bower does't work particularly well as root, so we create a non-root user
# called kwuser, but give it sudo ability.
RUN adduser --disabled-password --gecos '' kwuser && \
    adduser kwuser sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER kwuser

WORKDIR /home/kwuser

RUN mkdir -p /home/kwuser/.girder/logs

# Memex's proxy can't use the git protocol
#RUN git config --global url."https://".insteadOf git://

RUN git clone git://github.com/memex-explorer/geoinfo_space.git && \
    cd geoinfo_space && \
    git checkout optimizations && \
    git reset --hard

WORKDIR /home/kwuser/geoinfo_space

COPY makelocalcfg.py /home/kwuser/makelocalcfg.py

USER root

RUN python /home/kwuser/makelocalcfg.py > /etc/girder.cfg

USER kwuser

ENV KWDEMO_READY TRUE

CMD ["python", "src/main.py"]
