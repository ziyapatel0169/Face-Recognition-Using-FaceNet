
import os
import cv2
import numpy as np
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load saved embeddings
embeddings = np.load(
    os.path.join(BASE_DIR, "model_data", "embeddings.npy")
)

labels = np.load(
    os.path.join(BASE_DIR, "model_data", "labels.npy")
)

# Load pretrained FaceNet
embedder = FaceNet()

THRESHOLD = 0.70

# Start webcam
cap = cv2.VideoCapture(0)

print("Face Recognition Started!")
print("Press Q to exit")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera error!")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = embedder.extract(rgb, threshold=0.95)

    for face in faces:

        x, y, w, h = face["box"]

        new_embedding = face["embedding"].reshape(1, -1)

        similarities = cosine_similarity(
            new_embedding, embeddings
        )[0]

        best_index = np.argmax(similarities)
        best_similarity = similarities[best_index]

        if best_similarity >= THRESHOLD:
            name = str(labels[best_index])
            color = (0, 255, 0)
        else:
            name = "UNRECOGNIZED"
            color = (0, 0, 255)

        cv2.rectangle(
            frame, (x, y), (x + w, y + h), color, 2
        )

        cv2.putText(
            frame,
            f"{name} ({best_similarity:.2f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow("Face Recognition System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
