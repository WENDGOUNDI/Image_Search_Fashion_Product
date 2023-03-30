# Image Search Fashion Product

Image similarity search using deep learning is a technique used to find images that are visually similar to a given query image. This approach typically involves using a convolutional neural network (CNN) to extract features from the images, followed by a similarity metric to compare the features of the query image to those of a database of images. This technique is used in several domains in particular in Ecommerce to find similar items in a database. We tackle that implementation using a petrained model (VGG16) and a subset of a large Ecommerce dataset from kaggle.

# Dependencies
 - Numpy
 - Tensorflow
 - Keras
 - OS
 - Pickle
 - OpenCV
 - Matplotlib
 - Scikit-Learn
 - Streamlit
 - Pillow
 - Base64
 - IO
 
 # Dataset
 For this project we used the Fashion Product Images Dataset from kaggle. The dataset is large of 44.4k images with multiple category labels, descriptions and high-res images. In this project, we used a subset of this dataset, 1000 images for implementation.
 
 dataset link: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
 
 # Processes
 1. Load model and initialization
 2. Load/store Images as well as their paths.
 3. Create and save image embeddings in a pickle file.
 4. Use NearestNeighbors and cosine metric for distance calculation
 5. Display k similar products, k being the number of similar products. Here we fixed k=5

 
 # Result
 ![res3](https://user-images.githubusercontent.com/48753146/228759320-e0110817-a24b-4650-93c7-2dd67e713bb2.PNG)
 ![res1](https://user-images.githubusercontent.com/48753146/228759334-88648ee3-5e0d-444c-b056-c4b93774017f.PNG)
 ![res2](https://user-images.githubusercontent.com/48753146/228759341-8d7b9c02-c535-4be3-a232-769eb91f847a.PNG)
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
