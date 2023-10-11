import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import cv2


model_path = 'best_model1.h5'
model = tf.keras.models.load_model(model_path)

model_cnn = 'cnn_model.h5'
cnn_model = tf.keras.models.load_model(model_cnn)
custom_objects={'KerasLayer':hub.KerasLayer}


model_resnet = 'resnet_model.h5'
resnet_model = tf.keras.models.load_model(model_resnet, custom_objects=custom_objects)


emotion_classes = ['elephant','non_elephant']
def predict_emotion(image_file):
    img = Image.open(image_file)

    img = img.resize((256,256)) 
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_emotion = emotion_classes[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    return predicted_emotion, confidence

def predict_resnet(image_file):
    img = Image.open(image_file)
    img = np.array(img)
    # img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, size = [224, 224])
    img = tf.keras.applications.resnet.preprocess_input(img)
    # img = img/255.
    pred = resnet_model.predict(tf.expand_dims(img, axis=0))
    class_names = ['non_elephant','elephant']
    pred_class = class_names[int(tf.round(pred)[0][0])]
    return pred_class


def predict_cnn(image_file):
    classes=["non_elephant","elephant"]
    img = Image.open(image_file)
    image_array = np.array(img)
    imageSize = 50
    newImageArray = cv2.resize(image_array, (imageSize, imageSize))
    image = np.array(newImageArray, dtype="float32")
    image= np.expand_dims(image, axis=0)
    prediction  = cnn_model.predict(image)
    index = np.argmax(prediction[0],axis=0)
    return classes[index]



st.title("Elephant detection App")
st.write("Upload an image to predict")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    

    predicting_placeholder = st.empty()
    predicting_placeholder.write("Predicting...")

    predicted_emotion, confidence = predict_emotion(uploaded_file)
    predicted_resnet = predict_resnet(uploaded_file)
    predicted_cnn = predict_cnn(uploaded_file)
    predicting_placeholder.empty()
    st.subheader("Prediction Result inception v3")
    st.write(f"Predicted result: {predicted_emotion}")
    st.subheader("Prediction Result resnet")
    st.write(f"Predicted result: {predicted_resnet}")
    st.subheader("Prediction Result cnn")
    st.write(f"Predicted result: {predicted_cnn}")
    predictions = [predicted_emotion, predicted_cnn, predicted_resnet]
    classs = ['elephant', 'non_elephant']
    class_0_count = 0
    class_1_count = 0
    for prediction in predictions:
        if prediction == 'elephant': 
            class_0_count += 1
        elif prediction == 'non_elephant':
            class_1_count += 1

    # Determine the majority class
    if class_0_count > class_1_count:
        majority_class = 0
    else:
        majority_class = 1
    st.title("Final Prediction: "+classs[majority_class])
    
    








