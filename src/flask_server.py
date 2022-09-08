from flask import Flask
from emotions import EmotionDetector

app = Flask(__name__)
emotion_detector = EmotionDetector()
emotion_detector.start()

@app.route('/display/', methods=['GET', 'POST'])
def testest():
    #return "Hello World!"
    return "<div> " + str(emotion_detector.emotion_counters) + "</div>"

@app.route('/hello/', methods=['GET', 'POST'])
def welcome():
    return "Hello World!"

if  __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
