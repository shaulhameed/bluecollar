import hashlib, os
import getpass
import pyttsx

# import houndify
# import inspect
# from sample_wave import *


engine = pyttsx.init()
engine.setProperty('rate', 175)
voices = engine.getProperty('voices')
resource_file = "pwd.txt"
pwd_file = "gotyou.txt"


def encode(username, pwd):
    return "$%s::%s$" % (username, hashlib.sha512(pwd).hexdigest())


def add_user(username, pwd):
    if os.path.exists(resource_file):
        with open(resource_file) as f:
            if "$%s::" % username in f.read():
                raise Exception("User Already Exists")
    with open(resource_file, "a+") as f:
        print >> f, encode(username, pwd)
    return username


def check_login(username, pwd):
    with open(resource_file) as f:
        if encode(username, pwd) in f.read():
            return username


def create_username():
    # obj= MyListener();

    try:
        engine.say("Enter the user name and password for New user")
        engine.runAndWait()
        username = raw_input("Username:")
        password = open('gotyou.txt', 'r')
        pwd = password.readline()
        add_user(username, pwd)
        password.close()
        # username = add_user(raw_input("enter username:"), password=getpass.getpass())
        print("user credentials added")
        engine.say("User Credentials added! %s" % username)
        engine.runAndWait()
    except Exception as e:
        engine.say("recheck credentials")
        engine.runAndWait()


def login():
    engine.say("your username and password please")
    engine.runAndWait()
    username = (raw_input("Enter Username:"))
    password = open('gotyou.txt', 'r')
    pwd = password.readline()
    check_login(username, pwd)
    password.close()
    print("Login successful")
    engine.say("Welcoming You")
    engine.runAndWait()
    # unlock door here


engine.say("press c for new user and l for login")
engine.runAndWait()

while True:
    engine.say("choose New user or Login")
    {'c': create_username, 'l': login}.get(raw_input("(c)New user\n(l)Login\n>").lower(), login)()