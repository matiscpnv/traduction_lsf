import os
import cv2
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify, request
import tensorflow as tf
import numpy as np
import threading
import time

# Détection si on est sur Render (variable d’environnement)
IS_RENDER = os.environ.get("RENDER", "0") == "1"

# Initialisation de Flask
app = Flask(__name__)

# Initialisation MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Charger le modèle
model = tf.keras.models.load_model('sign_language_model.h5')

# Caméra (désactivée sur Render)
camera = None

if not IS_RENDER:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("❌ Erreur : Impossible d'ouvrir la caméra")
        exit()
else:
    print("🚫 Mode Render : Caméra désactivée")

# Variables globales
prediction = None
frame_count = 0
phrase = ""
last_letter = ""
last_time = time.time()

# Mapping ASL
label_map = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
    5: "F", 6: "G", 7: "H", 8: "I",
    9: "K", 10: "L", 11: "M", 12: "N", 13: "O",
    14: "P", 15: "Q", 16: "R", 17: "S", 18: "T",
    19: "U", 20: "V", 21: "W", 22: "X", 23: "Y"
}

# Predire la main
def predict_hand(frame):
    global prediction, phrase, last_letter, last_time

    hand_roi = frame[100:300, 100:300]
    gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized / 255.0
    input_image = normalized.reshape(1, 28, 28, 1)

    prediction = model.predict(input_image)
    predicted_class = int(np.argmax(prediction))
    letter = label_map.get(predicted_class, "")

    current_time = time.time()
    if letter == last_letter:
        if current_time - last_time > 1.0:
            phrase += letter
            last_letter = ""
            last_time = current_time
    else:
        last_letter = letter
        last_time = current_time

# Générateur vidéo
def gen_frames():
    global prediction, frame_count

    # Mode Render → pas de caméra
    if camera is None:
        while True:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Camera non disponible sur le serveur",
                        (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    # Mode local → webcam OK
    while True:
        success, frame = camera.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            frame_count += 1
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            if frame_count > 10:
                threading.Thread(target=predict_hand, args=(frame.copy(),)).start()
                frame_count = 0
        else:
            frame_count = 0

        if prediction is not None:
            predicted_class = int(np.argmax(prediction))
            letter = label_map.get(predicted_class, "?")
            cv2.putText(frame, f"Lettre : {letter}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Routes Flask
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['GET'])
def predict():
    global phrase
    return jsonify({"traduction": phrase})

@app.route('/reset', methods=['POST'])
def reset():
    global phrase
    phrase = ""
    return jsonify({"status": "ok"})

# Lancer
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000, debug=False)
