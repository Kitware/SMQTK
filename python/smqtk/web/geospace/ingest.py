import csv
import pymongo as pm
import time
import calendar

client = pm.MongoClient("localhost")
db = client.ist
db.drop_collection('ads2')

with open("istdata.txt", "r") as file:
    reader = csv.reader(file)
    time_pattern = '%Y-%m-%d %H:%M:%S'
    for row in reader:
        if len(row) >= 8 and (row[6] != "" and row[7] != ""):
            ad_time = row[4]
            if (ad_time is not None and ad_time != "None"):
                print ad_time
                ad_time = int(calendar.timegm(time.strptime(ad_time, time_pattern)))

            data = {"site":row[0], "field2":row[1], "field3":row[2], "place":row[3], "time":ad_time, "url":row[5], "loc":[float(row[7]), float(row[6])]}
            db.ads2.insert(data)



