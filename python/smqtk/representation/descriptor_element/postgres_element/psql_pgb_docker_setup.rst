Docker setup for Postgresql and PgBouncer
=========================================

This document will contain the docker container references and commands for
setting up and running a Postgres container with a linked PgBouncer container
with local postgres data-file storage.

This document should be treated as an example and not as a concrete way of
things.


Postgres Container and Setup
----------------------------

See official docker container documentation @ https://hub.docker.com/_/postgres/

.. prompt:: bash

    docker run \
        --name smqtk_postgres \
        -e "POSTGRES_USER=${POSTGRES_USER}" \
        -e "POSTGRES_PASSWORD=${POSTGRES_PASSWORD}" \
        -v /some/local/directory:/var/lib/postgresql/data \
        -p 5432:5432 \
        -d \
        postgres

Where `POSTGRES_USER` is set to some user roll to use and `POSTGRES_PASSWORD` is
set to a password for security.

The `-v` mount path can obviously be changed.
Setting an the additional env variable `PGDATA` would change where the postgres data is stored in the container, so the `-v` source path value would need to be the same.
Binding the data directory to a local path not only means data is kept around if the container ever goes away, but also allows us to modify the `postgresql.conf` file (sudo required).
We may then be able to tweak database settings and restart the container in order to customize the database.

The database port still needs to be proxied out of the container with the above `-p` option (<local>:<container> format)


PgBouncer Container and Setup
-----------------------------

Official documentation (https://hub.docker.com/r/mbentley/ubuntu-pgbouncer/) is
sparse on specific detail on how to setup the container, so I will detail
important steps here.

.. prompt:: bash

    sudo docker run \
        --name smqtk_pgb \
        -e "PG_ENV_POSTGRESQL_USER=${POSTGRES_USER}"
        -e "PG_ENV_POSTGRESQL_PASS=${POSTGRES_PASSWORD}" \
        -p 6432:6432 \
        --link ${LINK_NAME}:pg \
        -v /data/kitware/docker.pgbouncer.d/etc/pgbouncer:/etc/pgbouncer \
        -d \
        mbentley/ubuntu-pgbouncer

`POSTGRES_USER` and `POSTGRES_PASSWORD` used in the environment variables must
be the same as use in the Postgresql docker command above.

The port used needs to be exposed via the `-p` option.

The source postgres container name must be provided in the `--link` option (where the `${LINK_NAME}` var is).

The `-v` configuration files exposure is not particularly necessary, but is useful for debugging, if custom PgBouncer configuration settings are desired in the `pgbconf.ini` file, or if additional postgres rolls/passwords are desired in the `userlist.txt` file.
