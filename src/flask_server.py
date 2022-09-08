from flask import Flask
from emotions import EmotionDetector

app = Flask(__name__)
emotion_detector = EmotionDetector()
emotion_detector.start()


@app.route('/display/', methods=['GET', 'POST'])
def show_emotion_counters():
    return "<div> " + str(emotion_detector.emotion_counters) + "</div>"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
