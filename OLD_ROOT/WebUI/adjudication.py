"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""
"""
Blueprint for adjudication api
"""
import json
from bson import ObjectId
import flask
import datetime
from WebUI import db
from WebUI.utils import jsonify
from time import gmtime, strftime
import smtplib
from email.mime.text import MIMEText
import email

mod = flask.Blueprint('adjudication', __name__)

@mod.route('/', methods=['GET', 'POST'])
def adjudication():
    """
    Api for adjudication

    @return:
    """
    request = flask.request

    obj = {}

    # clip id
    clip = request.args.get("clip", None)
    if clip:
        obj["clip"] = clip

    # query : will be event kit in case of triage information
    query = request.args.get("query", None)
    if query:
        obj["query"] = query

    # If no label in the query then assume star, other options are yes and no
    obj["label"] = request.args.get("label", "star")

    # If no op in then assume add, other option is remove
    obj["op"] = request.args.get("op", "add")

    # Comment
    comment = request.args.get("comment", None)
    if comment:
        obj["comment"] = comment

    # Get ip address
    if not request.headers.getlist("X-Forwarded-For"):
       obj["ip"] = request.remote_addr
    else:
       obj["ip"] = request.headers.getlist("X-Forwarded-For")[0]

    # Get current time
    obj["time"] = datetime.datetime.now()
    obj["strtime"] = str(obj["time"])

    # get user
    obj["user"] = "anonymous"

    if flask.session.has_key("user"):
        # user logged in
        obj["user"] = flask.session["user"]["fullname"]

    db["adjudication"].insert(obj)

    return jsonify(obj)

# Stuff for mailing

def make_csv(acts, user):
    csvtext = "SessionStartDate,SessionStartTime,JudgeID,EventID,EventKitVideos,ClipID,ClipRank,Decision,ElaspedTimeFromStartOfEventIDTriage\n"
    for anact in acts:
        adj = "IsEvent"
        if anact["label"] == "star" :
            adj = "Unsure"
        elif anact["label"] == "no" :
            adj = "IsNotEvent"

        actstr = anact["time"] + \
                 user + "," + \
                 "E" + str(anact["query"]["event"]).zfill(3) + "," + \
                 anact["query"]["training"] + "," + \
                 anact["clip"][3:] + ","  + \
                 str(anact["rank"]) + "," + \
                 adj + "," + \
                 str(anact["elapsed"]) + "\n"

        csvtext = csvtext + actstr

    return csvtext

@mod.route('/mail', methods=['POST'])
def email_adjudications():
    obj = {}
    obj["user"] = flask.session["user"]
    obj["user_activity"] = []
    # Create email
    if flask.request.method == "POST":
        obj["user_activity"] = json.loads(flask.request.form.get("user_activity", "[]"))

    obj["email"] = "not yet"
    emailfrom  = "SOME-ADDRESS@foo.com"  # TODO
    #emailto = "dhandeo@gmail.com, megha.pandey@kitware.com"
    emailto = "dhandeo@gmail.com, sangmin.oh@kitware.com, megha.pandey@kitware.com"

    body = MIMEText("Hello David" + ",\n\n"
        + "Please find the triage spreadsheet as attached- \n\n"
        + "\nThank you,\nThe Kitware SMQTK Team\n")

    # Create a text/plain message
    msg = email.MIMEMultipart.MIMEMultipart('alternative')

    attachment = MIMEText(make_csv(obj["user_activity"], flask.session["user"]["fullname"]))
    fname = flask.session["user"]["fullname"].replace(" ", "")
    fname = fname + "_E" + str(int(obj["user_activity"][0]["query"]["event"])).zfill(3)
    fname = fname + "_" + obj["user_activity"][0]["query"]["training"]
    fname = fname + "_eval.csv"

    obj["fname"] = fname
    attachment.add_header('Content-Disposition', 'attachment', filename=fname)
    msg.attach(attachment)

    # me == the sender's email address
    # you == the recipient's email address
    msg['Subject'] = 'SMQTK Evaluation for Triage at ' + strftime("%Y-%m-%d %H:%M:%S", gmtime())
    msg['From'] = emailfrom
    msg['To'] = emailto
    msg.attach(body)

    # TODO: Would send email here to some address specified above
    # s = smtplib.SMTP("SOME-ADDRESS")
    # try:
    #     out = s.sendmail(emailfrom, emailto.split(","), msg.as_string())
    # except Exception as e:
    #     obj["error"] = str(type(e)) + ": " + str(e)
    #     return jsonify(obj)

    return flask.jsonify(obj)

