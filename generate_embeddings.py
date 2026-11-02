
import os
import numpy as np
from PIL import Image
from keras_facenet import FaceNet

# Load FaceNet model
print("Loading FaceNet...")
embedder = FaceNet()

# Dataset path

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "train"
)
# Store embeddings and labels
embeddings = []
labels = []

# Counters
total = 0
detected = 0
skipped = 0
errors = 0

print("\nStarting Face Recognition Preprocessing...\n")

# Process each person
for person in sorted(os.listdir(DATASET_PATH)):

    person_path = os.path.join(DATASET_PATH, person)

    if not os.path.isdir(person_path):
        continue

    p_total = 0
    p_detected = 0
    p_skipped = 0
    p_errors = 0

    print(f"\n========== {person} ==========")

    for filename in sorted(os.listdir(person_path)):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        total += 1
        p_total += 1

        image_path = os.path.join(person_path, filename)

        try:
            # Read image
            image = np.array(
                Image.open(image_path).convert("RGB")
            )

            # Detect face and generate embedding
            faces = embedder.extract(
                image,
                threshold=0.95
            )

            # No face detected
            if len(faces) == 0:

                print(f"NO FACE: {filename}")

                skipped += 1
                p_skipped += 1
                continue

            # Multiple faces detected
            if len(faces) > 1:

                print(
                    f"MULTIPLE FACES: {filename} "
                    f"({len(faces)} faces)"
                )

                skipped += 1
                p_skipped += 1
                continue

            # Exactly one face detected
            embedding = faces[0]["embedding"]

            embeddings.append(embedding)
            labels.append(person)

            detected += 1
            p_detected += 1

            print(f"ACCEPTED: {filename}")

        except Exception as e:

            print(f"ERROR: {filename} | {e}")

            errors += 1
            p_errors += 1

    # Person-wise report
    print(f"\n{person} REPORT")
    print("Total:", p_total)
    print("Detected:", p_detected)
    print("Skipped:", p_skipped)
    print("Errors:", p_errors)

# Create output directory

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model_data"
)
os.makedirs(MODEL_DIR, exist_ok=True)

# Convert to arrays
embeddings = np.array(embeddings)
labels = np.array(labels)

# Save files
if len(embeddings) > 0:

    np.save(
        os.path.join(MODEL_DIR, "embeddings.npy"),
        embeddings
    )

    np.save(
        os.path.join(MODEL_DIR, "labels.npy"),
        labels
    )

# Final report
print("\n================================")
print("       FINAL DATASET REPORT")
print("================================")

print("Total images:", total)
print("Successfully detected:", detected)
print("Skipped images:", skipped)
print("Processing errors:", errors)

print("\nEmbedding shape:", embeddings.shape)

print("\nPerson-wise distribution:")

for person in sorted(set(labels)):
    count = np.sum(labels == person)
    print(f"{person}: {count} embeddings")

print("\n================================")

if detected > 0:
    print("EMBEDDINGS SAVED SUCCESSFULLY!")
else:
    print("NO EMBEDDINGS GENERATED. CHECK DATASET!")

print("================================")
