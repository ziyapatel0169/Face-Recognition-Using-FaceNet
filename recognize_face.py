
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load saved embeddings
embeddings = np.load(
    os.path.join(BASE_DIR, "model_data", "embeddings.npy")
)

labels = np.load(
    os.path.join(BASE_DIR, "model_data", "labels.npy")
)


def recognize_face(new_embedding):

    new_embedding = new_embedding.reshape(1, -1)

    similarities = cosine_similarity(
        new_embedding, embeddings
    )[0]

    best_index = np.argmax(similarities)
    best_similarity = similarities[best_index]

    predicted_name = labels[best_index]

    THRESHOLD = 0.70

    if best_similarity >= THRESHOLD:
        return predicted_name, best_similarity

    return "UNRECOGNIZED", best_similarity


print("Recognition system loaded!")
print("Registered people:", np.unique(labels))
