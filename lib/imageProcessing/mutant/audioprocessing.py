import pyttsx
class alert:

    def detect_fail(self):
        engine = pyttsx.init()
        engine.say('i can not see you,please be in your position')
        engine.runAndWait()
    def pwd(self):
        engine = pyttsx.init()
        engine.say('Your password please!')
        engine.runAndWait()