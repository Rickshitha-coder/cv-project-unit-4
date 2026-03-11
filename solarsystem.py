import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Solar System AR Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png"])

reference = cv2.imread("solar.jpg",0)

if uploaded_file:

    image = Image.open(uploaded_file)
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(1000)

    kp1, des1 = orb.detectAndCompute(reference,None)
    kp2, des2 = orb.detectAndCompute(gray,None)

    if des1 is None or des2 is None:
        st.error("Features not detected")
    else:

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        matches = bf.match(des1,des2)

        st.write("Matching Features:",len(matches))

        if len(matches) > 10:

            st.success("Solar System Detected")

            st.write("Mercury")
            st.write("Venus")
            st.write("Earth")
            st.write("Mars")
            st.write("Jupiter")
            st.write("Saturn")
            st.write("Uranus")
            st.write("Neptune")

        else:
            st.error("Solar System image not detected")
