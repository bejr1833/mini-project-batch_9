import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import matplotlib.pyplot as plt

# Paths to your local files
CSV_PATH = "C:/Users/josth/OneDrive/Desktop/face recognition/backend/dataset/Updated_Dataset.csv"
FACES_FOLDER = "C:/Users/josth/OneDrive/Desktop/face recognition/backend/dataset/Facial/Faces"
EMBEDDINGS_PATH = "C:/Users/josth/OneDrive/Desktop/face recognition/backend/embeddings/face_embeddings.npz"

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Function to extract embedding from an image
def extract_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    face = mtcnn(image)
    if face is None:
        return None
    with torch.no_grad():
        embedding = resnet(face.unsqueeze(0))
    return embedding[0].cpu().numpy()

# Function to find the most similar person
def find_similar_persononin(image_path, threshold=0.6):
    # Extract embedding of uploaded image
    embedding = extract_embedding(image_path)
    if embedding is None:
        print("❌ No face detected in the image.")
        return

    # 🔄 Reload the latest embeddings and labels
    data = np.load(EMBEDDINGS_PATH)
    embeddings = data["embeddings"]
    labels = data["labels"]

    # Refit Nearest Neighbors
    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', n_jobs=-1)
    nn.fit(embeddings)

    # Read CSV again to get the latest details
    dataset = pd.read_csv(CSV_PATH)

    # Perform matching
    distances, indices = nn.kneighbors([embedding])
    min_distance = distances[0][0]
    matched_index = indices[0][0]
    predicted_label = labels[matched_index]

    print(f"Predicted Label: {predicted_label}")  # Debug print

    # Check if predicted_label exists in dataset
    if predicted_label not in dataset['label'].values:
        print(f"❌ Label {predicted_label} not found in dataset.")
        return

    matched_details = dataset[dataset['label'] == predicted_label].iloc[0]
    matched_image_path = f"{FACES_FOLDER}/{matched_details['id']}"

    person_id = matched_details["id"]
    last_seen = matched_details["Last Seen Location"]
    missing_since = matched_details["Missing Since"]
    contact = matched_details["Mobile Number"]

    input_img = Image.open(image_path)
    matched_img = Image.open(matched_image_path)
    
    # Display the images
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(input_img)
    ax[0].set_title("Uploaded Image")
    ax[0].axis("off")

    ax[1].imshow(matched_img)
    ax[1].set_title(f"Most Similar: {predicted_label}")
    ax[1].axis("off")
    plt.show()

    if min_distance < threshold:
        print(f"✅ Match Found: {predicted_label}")
    else:
        print("❌ No exact match found. Showing the most similar match.")

    print(f"\n🆔 ID: {person_id}")
    print(f"👤 Name: {predicted_label}")
    print(f"📍 Last Seen: {last_seen}")
    print(f"📅 Missing Since: {missing_since}")
    print(f"📞 Contact: {contact}")

test_img = "C:\\Users\\josth\\OneDrive\\Desktop\\face recognition\\backend\\dataset\\Facial\\Faces\\Akshay Kumar_0.jpg"


find_similar_persononin(test_img)

