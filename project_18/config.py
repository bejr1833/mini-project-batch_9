import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "supersecretkey")
    DEBUG = True

    # Local paths for your project
    EMBEDDINGS_PATH = "C:/Users/josth/OneDrive/Desktop/face recognition/backend/embeddings/face_embeddings.npz"
    FACES_FOLDER = "C:/Users/josth/OneDrive/Desktop/face recognition/backend/dataset/Facial/Faces"
    CSV_PATH = "C:/Users/josth/OneDrive/Desktop/face recognition/backend/dataset/Updated_Dataset.csv"
    UPLOAD_FOLDER = "C:/Users/josth/OneDrive/Desktop/face recognition/backend/uploads"

    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
