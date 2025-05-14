from PIL import Image
import numpy as np
import pandas as pd
import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# Update local paths
EMBEDDINGS_PATH = "C:/Users/josth/OneDrive/Desktop/face recognition/backend/embeddings/face_embeddings.npz"
SAVE_FOLDER = "C:/Users/josth/OneDrive/Desktop/face recognition/backend/dataset/Facial/Faces"
CSV_PATH = "C:/Users/josth/OneDrive/Desktop/face recognition/backend/dataset/Updated_Dataset.csv"

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=False, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def extract_face_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    face = mtcnn(image)
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
