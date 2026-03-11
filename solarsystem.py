import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="AR Solar System App", page_icon="🌍")

st.title("🌌 AR Educational App - Solar System")
st.write("Upload a Solar System image to see planet information.")

# Upload image
uploaded_file = st.file_uploader("Upload Solar System Image", type=["jpg", "png"])

# Reference image
reference = cv2.imread("solar_system.jpg", 0)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ORB feature detection
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(reference, None)
    kp2, des2 = orb.detectAndCompute(gray, None)

    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    st.write("Matching Features:", len(matches))

    # If enough matches found
    if len(matches) > 20:

        st.success("Solar System Image Detected! 🌞")

        st.subheader("Planets Information")

        st.write("☿ Mercury – Closest planet to the Sun")
        st.write("♀ Venus – Hottest planet")
        st.write("🌍 Earth – Our home planet")
        st.write("♂ Mars – Known as the Red Planet")
        st.write("♃ Jupiter – Largest planet")
        st.write("♄ Saturn – Has beautiful rings")
        st.write("♅ Uranus – Ice giant")
        st.write("♆ Neptune – Farthest planet")

    else:
        st.error("Solar System image not detected")