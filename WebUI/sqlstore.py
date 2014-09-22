"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""
import sqlite3
import sys
import os

from WebUI import app


class datastore:

    def __init__(self, path):
        self.data_file = path

    def connect(self):
        self.conn = sqlite3.connect(self.data_file)
        return self.conn.cursor()

    def disconnect(self):
        self.cursor.close()

    def free(self, cursor):
        cursor.close()

    def write(self, query, values = ''):
        cursor = self.connect()
        if values != '':
            cursor.execute(query, values)
        else:
            cursor.execute(query)
        self.conn.commit()
        return cursor

    def read(self, query, values = ''):
        cursor = self.connect()
        if values != '':
            cursor.execute(query, values)
        else:
            cursor.execute(query)
        return cursor


medtest_sqlstore = datastore(os.path.join(app.config['STATIC_DIR'], 'data/clip_calib_medtest.sqlite'))
eventkits_sqlstore = datastore(os.path.join(app.config['STATIC_DIR'], 'data/clip_calib_eventkits.sqlite'))


"""
'SELECT ob.v_id, ob.small_group_of_people, ob.large_group_of_people, MAX(ob.small_group_of_people,ob.large_group_of_people) as result FROM ob, sc WHERE ob.v_id = sc.v_id;

"""
