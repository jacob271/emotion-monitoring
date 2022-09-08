from flask import Flask
from emotion_monitoring import EmotionMonitor

app = Flask(__name__)
emotion_monitor = EmotionMonitor()
emotion_monitor.start()


@app.route('/display/', methods=['GET', 'POST'])
def show_emotion_counters():
    return "<div> " + str(emotion_monitor.emotion_counters) + "</div>"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
