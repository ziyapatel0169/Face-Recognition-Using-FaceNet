
import os
import base64
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from keras_facenet import FaceNet


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)


def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    return np.dot(a, b.T) / (
        np.linalg.norm(a, axis=1, keepdims=True) *
        np.linalg.norm(b, axis=1)
    )


# Load saved embeddings and labels
embeddings = np.load(
    os.path.join(BASE_DIR, "model_data", "embeddings.npy")
)

labels = np.load(
    os.path.join(BASE_DIR, "model_data", "labels.npy")
)

# Load pretrained FaceNet
embedder = FaceNet()

THRESHOLD = 0.70

history = []
registered_people = sorted(set(labels.tolist()))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/people")
def people():
    return jsonify(registered_people)


@app.route("/history")
def get_history():
    return jsonify(history)

@app.route("/clear-history", methods=["POST"])
def clear_history():
    history.clear()
    return jsonify({"message": "History cleared"})

@app.route("/recognize", methods=["POST"])
def recognize():
    try:
        data = request.json["image"]
        image_data = base64.b64decode(data.split(",")[1])

        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid image"}), 400

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = embedder.extract(rgb, threshold=0.95)

        if len(faces) == 0:
            return jsonify({
                "name": "NO FACE",
                "status": "No face detected"
            })

        if len(faces) > 1:
            return jsonify({
                "name": "MULTIPLE FACES",
                "status": "Multiple faces detected"
            })

        new_embedding = faces[0]["embedding"].reshape(1, -1)

        similarities = cosine_similarity(
            new_embedding, embeddings
        )[0]

        best_index = np.argmax(similarities)
        score = float(similarities[best_index])

        if score >= THRESHOLD:
            name = str(labels[best_index])
            status = "Recognized"
        else:
            name = "UNRECOGNIZED"
            status = "Unknown person"

        timestamp = datetime.now().strftime(
            "%d-%m-%Y %I:%M:%S %p"
        )

        event = {
            "name": name,
            "status": status,
            "confidence": round(score, 3),
            "time": timestamp
        }

        if not history or history[-1]["name"] != name:
            history.insert(0, event)
            del history[50:]

        return jsonify(event)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False)
