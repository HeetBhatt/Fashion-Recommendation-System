import os
from PIL import Image
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import streamlit as st  
import numpy as np
from numpy.linalg import norm
import pickle
from sklearn.neighbors import NearestNeighbors

st.title("Fashion Recommendation System")

feature_list = pickle.load(open("embedding.pkl","rb"))
filenames = pickle.load(open("filenames.pkl","rb"))
# model
model = ResNet50(weights = 'imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False

#adding new layer 
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join("uploaded",uploaded_file.name),"wb") as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    #converting image to numpy array
    img_array = image.img_to_array(img)
    #reshaping the array to (1,224,224,3) dimentions because keras works batch wise,it doesn't work on single image.
    expanded_img_array = np.expand_dims(img_array,axis = 0)
    #now we need to convert input to a correct format so here I have used preprocess_input
    #1st change is images are converted to RGB to BGR
    #2nd change is each color channel is zero centered with respect to the imagenet dataset without scalling
    preprocessing_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessing_img).flatten()
    #deviding the result by norm of matrix
    normalized_result = result/norm(result)
    return normalized_result

def recommend(features,feature_list):
    neighbours = NearestNeighbors(n_neighbors=6,algorithm="brute",metric="euclidean")
    neighbours.fit(feature_list)
    distances,indices=neighbours.kneighbors([features])
    return indices

    

# file upload
uploaded_file = st.file_uploader("choose an image")
if(uploaded_file is not None):
    if(save_uploaded_file(uploaded_file)):
        #display image
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        #feature extract
        features=feature_extraction(os.path.join("uploaded",uploaded_file.name),model)
        # st.text(features)
        #recommendation
        indices = recommend(features,feature_list)   
        #show
        column1,column2,column3,column4,column5 = st.columns(5)
        with column1:
            st.image(filenames[indices[0][1]])
        with column2:
            st.image(filenames[indices[0][2]])
        with column3:
            st.image(filenames[indices[0][3]])
        with column4:
            st.image(filenames[indices[0][4]])
        with column5:
            st.image(filenames[indices[0][5]])
    else:
        st.header("some error occured in file upload")
   



# python -m virtualenv .
# .\scripts\activate
# virtualenv streamlitenv