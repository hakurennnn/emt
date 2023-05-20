import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

@st.cache(allow_output_mutation=True)
def load_weather_model():
    model = load_model('modelsaved.h5')
    return model

def preprocess_image(image):
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def classify_weather(image):
    model = load_weather_model()
    img = preprocess_image(image)
    prediction = model.predict(img)
    labels = ['cloudy', 'rainy', 'shine', 'sunrise']
    predicted_label = labels[np.argmax(prediction)]
    return predicted_label

def main():
    st.title("Weather Classification")
    st.write("Upload an image and let me classify the weather!")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        predicted_weather = classify_weather(image)

        st.write("Predicted Weather:", predicted_weather)

if __name__ == '__main__':
    main()
