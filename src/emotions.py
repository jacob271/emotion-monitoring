import numpy as np
import cv2
import subprocess
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import os
import time
import threading
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class EmotionDetector(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

        # Create the model
        self.model = Sequential()

        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(7, activation='softmax'))
        self.emotion_counters = [0, 0, 0, 0, 0, 0, 0]
        self.count = 1

    def run(self):
        self.model.load_weights('model.h5')
        # prevents openCL usage and unnecessary logging messages
        cv2.ocl.setUseOpenCL(False)
        # dictionary which assigns each label an emotion (alphabetical order)
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fatigue", 3: "Happy", 4: "Neutral", 5: "Fatigue",
                        6: "Surprised"}

        # start the webcam feed
        cap = cv2.VideoCapture(0)
        notification_interval = 5 * 1000
        self.count = 1
        input()
        start_time = round(time.time() * 1000)
        while True:
            # Find haar cascade to draw bounding box around face
            ret, frame = cap.read()
            if not ret:
                break
            facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                self.count += 1
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = self.model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)
                self.emotion_counters[maxindex] += 1
            if round(time.time() * 1000) - start_time >= notification_interval and self.count > 0:
                print("Emotion counters: " + str(self.emotion_counters))
                print("Counter: " + str(self.count))
                if self.emotion_counters[3] / self.count > 0.4:
                    print("Happy percentage: " + str(self.emotion_counters[3] / self.count))
                    matplotlib.rcParams['toolbar'] = 'None'
                    img = mpimg.imread('happy.png')
                    plt.imshow(img)
                    plt.axis('off')
                    plt.tight_layout()
                    plt.get_current_fig_manager().canvas.set_window_title('pfxt')
                    plt.show()
                elif (self.emotion_counters[5] + self.emotion_counters[2]) / self.count > 0.2:
                    print(
                        "fatigue percentage" + str((self.emotion_counters[5] + self.emotion_counters[2]) / self.count))
                    matplotlib.rcParams['toolbar'] = 'None'
                    img = mpimg.imread('tired.png')
                    plt.imshow(img)
                    plt.axis('off')
                    plt.tight_layout()
                    plt.get_current_fig_manager().canvas.set_window_title('pfxt')
                    plt.show()
                else:
                    print("not sleepy or happy")
                self.count = 0
                start_time = round(time.time() * 1000)
                for i in range(0, 6):
                    self.emotion_counters[i] = 0
            cv2.imshow('Video', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    emotion_detection = EmotionDetector()
    emotion_detection.start()
