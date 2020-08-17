import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pytesseract
import cv2
import os

st.set_option('deprecation.showfileUploaderEncoding', False)

#pytesseract.pytesseract.tesseract_cmd = r"F:\Program Files\Tesseract-OCR\tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = '/app/.apt/usr/bin/tesseract'

def detect_ocr(image):
    # img = Image.open(image)
    img = np.array(image.convert('RGB'))
    text = pytesseract.image_to_string(img, lang='eng')
    return img, text


try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
except Exception:
    st.write("Error loading cascade classifiers")


def detect_img(image):
    image = np.array(image.convert('RGB'))

    faces = face_cascade.detectMultiScale(image=image, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img=image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
        roi = image[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi)
        smile = smile_cascade.detectMultiScale(roi, minNeighbors=25)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

    return image, faces


def face_rec():
    st.write('''**Face Recognition**

Detect the number of faces from the uploaded picture.
        ''')
    image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])
    st.markdown('''
            <style> body{ 
                    color: #565555;
                    background-color: #30555a87;
                }</style>''',unsafe_allow_html=True)

    if image_file is not None:
        image = Image.open(image_file)
        if st.button("Process"):
            result_img, result_faces = detect_img(image=image)
            st.image(result_img, use_column_width=True)
            st.success("Found {} faces\n".format(len(result_faces)))


def text_rec():
    st.write("Convert image to text using pytesseract")
    image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])
    st.markdown('''
            <style> body{ 
                        color: #5a5759;
                        background-color: #5fc1bb;
                }</style>''',unsafe_allow_html=True)

    if image_file is not None:

        image = Image.open(image_file)

        if st.button("Process"):
            final_img, text = detect_ocr(image=image)
            st.image(final_img, use_column_width=True)
            st.write("Text found in the image")
            st.success("\n {} \n".format(text))


def about():

    st.write(
        '''
        **Haar Cascade** is an object detection algorithm.

_It can be used to detect objects in images or videos. _
_The algorithm has four stages: _

1. Haar Feature Selection 
2. Creating  Integral Images
3. Adaboost Training 
4. Cascading Classifiers 

Read more : https://github.com/zatch3301
        ''')
    st.markdown('''
            <style> body{ 
                        color: #f3f0f0;
                        background-color: #3c3c3c;
                }</style>''',unsafe_allow_html=True)

image = Image.open('bg.png')

def main():
    st.title("Half-Exploit :octopus:")  
    stt.set_theme({'primary': '#32a877'})   
    activities = ["Home", "Face Recognition", "Img OCR", "About"]
    choice = st.sidebar.selectbox("Modules", activities)
    hide_streamlit_style = """
            <title> Half Explot </title>
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .sidebar .sidebar-content {background-image: linear-gradient(180deg,#4CA1AF,#2c3e50);}
            .btn-outline-secondary {
            border-color: #09ab3b85;
            color: #f9f9f9;
            }
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

    if choice == "Home":
        st.write("**by _zatch_**")
        st.markdown('''<h3 style align = "center"><i>Welcome to Hexa-Exploit one solution for all</i></h3>
            <style> body{ 
                    color: #efefef;
                    background-color: #676a75;
                }
                </style>''',unsafe_allow_html=True)
        st.image(image, caption='Sunrise by the mountains',use_column_width=True)
        st.write("Go to the About section from the sidebar to learn more about it.")
    elif choice == "Face Recognition":
        face_rec()
    elif choice == "Img OCR":
        text_rec()
    elif choice == "About":
        about()


if __name__ == '__main__':
    main()
