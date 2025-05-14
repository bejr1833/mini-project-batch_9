from PIL import Image
import numpy as np
import pandas as pd
import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
# Update local paths
EMBEDDINGS_PATH = "C:/Users/josth/OneDrive/Desktop/face recognition/backend/embeddings/face_embeddings.npz"
SAVE_FOLDER = "C:/Users/josth/OneDrive/Desktop/face recognition/backend/dataset/Facial/Faces"
CSV_PATH = "C:/Users/josth/OneDrive/Desktop/face recognition/backend/dataset/Updated_Dataset.csv"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn_add = MTCNN(keep_all=False, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def extract_face_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    face = mtcnn_add(image)
    if face is None:
        return None
    with torch.no_grad():
        embedding = model(face.unsqueeze(0))
    return embedding[0].cpu().numpy()

def add_new_image(image_path, person_id, name, last_seen, missing_since, phone):
    embedding = extract_face_embedding(image_path)
    if embedding is None:
        return "❌ Face not detected. Please try another image."

    # Load existing embeddings
    if os.path.exists(EMBEDDINGS_PATH):
        data = np.load(EMBEDDINGS_PATH)
        embeddings_db = data["embeddings"]
        labels_db = data["labels"]
    else:
        embeddings_db = np.empty((0, 512))
        labels_db = np.empty((0,), dtype='<U50')

    # Add the new embedding
    embeddings_db = np.vstack([embeddings_db, embedding])
    labels_db = np.append(labels_db, name)
    np.savez(EMBEDDINGS_PATH, embeddings=embeddings_db, labels=labels_db)

    # Save image
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    image_save_path = os.path.join(SAVE_FOLDER, person_id)
    Image.open(image_path).save(image_save_path)

    # Update CSV
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        df = pd.DataFrame(columns=['id', 'label', 'Image Path', 'Last Seen Location', 'Missing Since', 'Mobile Number'])

    new_entry = {
        'id': person_id,
        'label': name,
        'Image Path': image_save_path,
        'Last Seen Location': last_seen,
        'Missing Since': missing_since,
        'Mobile Number': phone
    }

    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)

    return f"✅ Added {name} ({person_id}) successfully!"


# face_matcher.py

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import io
import base64

# Load models once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

EMBEDDINGS_PATH = "C:/Users/josth/OneDrive/Desktop/face recognition/backend/embeddings/face_embeddings.npz"
SAVE_FOLDER = "C:/Users/josth/OneDrive/Desktop/face recognition/backend/dataset/Facial/Faces"
CSV_PATH = "C:/Users/josth/OneDrive/Desktop/face recognition/backend/dataset/Updated_Dataset.csv"


def extract_embedding(image):
    face = mtcnn(image)
    if face is None:
        return None
    with torch.no_grad():
        embedding = resnet(face.unsqueeze(0))
    return embedding[0].cpu().numpy()

def find_similar_person_from_file(uploaded_file, threshold=0.6):
    image = Image.open(uploaded_file).convert("RGB")
    embedding = extract_embedding(image)

    if embedding is None:
        return {"error": "No face detected in the uploaded image."}

    data = np.load(EMBEDDINGS_PATH)
    embeddings = data["embeddings"]
    labels = data["labels"]

    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    nn.fit(embeddings)

    dataset = pd.read_csv(CSV_PATH)

    distances, indices = nn.kneighbors([embedding])
    min_distance = distances[0][0]
    matched_index = indices[0][0]
    predicted_label = labels[matched_index]

    if predicted_label not in dataset['label'].values:
        return {"error": f"Label {predicted_label} not found in dataset."}

    matched_details = dataset[dataset['label'] == predicted_label].iloc[0]
    matched_image_path = f"{SAVE_FOLDER}/{matched_details['id']}"

    # Convert matched image to base64
    with open(matched_image_path, "rb") as img_file:
        encoded_matched_image = base64.b64encode(img_file.read()).decode()

    # Convert uploaded image to base64
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    encoded_uploaded_image = base64.b64encode(buffer.getvalue()).decode()

    return {
        "match_found": min_distance < threshold,
        "distance": float(min_distance),
        "name": predicted_label,
        "person_id": matched_details["id"],
        "last_seen": matched_details["Last Seen Location"],
        "missing_since": matched_details["Missing Since"],
        "contact": matched_details["Mobile Number"],
        "image_base64": encoded_matched_image,
        "uploaded_image": encoded_uploaded_image
    }
