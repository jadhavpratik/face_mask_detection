from flask import Flask, render_template, Response
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

app = Flask(__name__)
model = load_model("mask_detector.model")


def detect_mask(frame):
    image = cv2.resize(frame, (224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    preds = model.predict(image)
    label = "With Mask" if preds[0][0] > preds[0][1] else "Without Mask"
    color = (0, 255, 0) if label == "With Mask" else (255, 0, 0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]
        roi = cv2.resize(roi, (224, 224))
        roi = img_to_array(roi)
        roi = preprocess_input(roi)
        roi = np.expand_dims(roi, axis=0)

        face_preds = model.predict(roi)
        face_label = "With Mask" if face_preds[0][0] > face_preds[0][1] else "Without Mask"
        face_color = (0, 255, 0) if face_label == "With Mask" else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), face_color, 2)
        cv2.putText(frame, face_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 2)

    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return frame

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = detect_mask(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
