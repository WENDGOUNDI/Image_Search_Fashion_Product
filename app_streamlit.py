import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import cv2
import os
import pickle
from PIL import Image
from io import BytesIO
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import base64

with open("logo.png", "rb") as f:
    data = base64.b64encode(f.read()).decode("utf-8")

    st.sidebar.markdown(
        f"""
        <div style="display:table;margin-top:-20%;margin-left:20%;">
            <img src="data:image/png;base64,{data}" width="100" height="150">
        </div>
        """,
        unsafe_allow_html=True,
    )

st.title("ECOMMERCE PRODUCT SIMILARITY SEARCH WEBAPP")


# Load VGG16 pretrained model
model = VGG16(weights='imagenet', include_top=False)

# Function to extract the features from an image using the VGG16 model
def extract_features(img_path):
    img = img_path.resize((224, 224))
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    features = features.flatten()
    return features

# Load embeddings saved as pickle file
with open('database_dl_images_embeddings.pickle', 'rb') as handle:
    data_embeddings = pickle.load(handle)

# Initialize the embeddings and images paths variables
nump_features = list(data_embeddings.values())
images_path = list(data_embeddings.keys())


# Build NearestNeighbors model and fit to features
nn_model = NearestNeighbors(n_neighbors=6, metric='cosine')
nn_model.fit(nump_features)

# Function to perform image similar search
def search_similar_images(query_path, image_paths):
    # Extract features from query image
    query_features = extract_features(query_path)

    # Find most similar images using NearestNeighbors model
    distances, indices = nn_model.kneighbors([query_features])
    similar_images = [image_paths[i] for i in indices.flatten()]

    return similar_images


# LOAD IMAGE
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["png","jpg", "jpeg"])    


if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    img = Image.open(BytesIO(bytes_data))
    img = img.convert('RGB')
    display_img = img.resize((150,150))
    st.image(display_img)


if st.button('Search'):
    # Example usage
    query_path = bytearray(uploaded_file.getvalue())
    similar_images = search_similar_images(img, images_path)
    

    # Read results  
    img1 = plt.imread(similar_images[1])
    img2 = plt.imread(similar_images[2])
    img3 = plt.imread(similar_images[3])
    img4 = plt.imread(similar_images[4])
    img5 = plt.imread(similar_images[5])

    st.header(" Displayed from the most similar to the least one")
    st.image([img1, img2, img3, img4, img5], width=125)
    