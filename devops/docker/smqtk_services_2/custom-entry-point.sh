#!/usr/bin/env bash

PSQL="psql -h smqtk-postgres -U postgres"

# WAIT FOR DB SERVER
trigger="DatabaseNowResponsive"
poll_command="$PSQL -c \"\\\\echo $trigger\" 2>/dev/null | grep -q \"$trigger\""

echo "Waiting for a responsive database"

eval "${poll_command}"
while (("$?")) ; do eval "{$poll_command}" ; done

# CREATE TABLES IN DB
lock_dir=/tables-created.lock

if [ '!' -d "$lock_dir" ] ; then
    ${PSQL} << EOSQL
$(cat \
  /smqtk/install/etc/smqtk/postgres/descriptor_element/example_table_init.sql \
  /smqtk/install/etc/smqtk/postgres/descriptor_index/example_table_init.sql)
EOSQL

    mkdir "$lock_dir"
fi

exec /app/scripts/entrypoint.sh "$@"
