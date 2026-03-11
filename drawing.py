import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🎨 Image Cartoon Filter App")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    img = np.array(image)

    st.subheader("Original Image")
    st.image(img)

    option = st.selectbox(
        "Choose Filter",
        ("Cartoon", "Pencil Sketch", "Edge Detection")
    )

    if option == "Cartoon":

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray,5)

        edges = cv2.adaptiveThreshold(
            gray,255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,9,9
        )

        color = cv2.bilateralFilter(img,9,250,250)

        cartoon = cv2.bitwise_and(color,color,mask=edges)

        st.subheader("Cartoon Image")
        st.image(cartoon)

    elif option == "Pencil Sketch":

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        invert = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(invert,(21,21),0)
        inverted_blur = cv2.bitwise_not(blur)

        sketch = cv2.divide(gray,inverted_blur,scale=256.0)

        st.subheader("Pencil Sketch")
        st.image(sketch)

    elif option == "Edge Detection":

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,100,200)

        st.subheader("Edge Detection")
        st.image(edges)
