import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Page config
st.set_page_config(
    page_title="🎨 Cartoon Filter App",
    page_icon="🎨",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
.big-title{
    font-size:40px;
    font-weight:bold;
    text-align:center;
    color:#ff4b4b;
}

.subtitle{
    text-align:center;
    color:gray;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🎨 Image Cartoon Filter App</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Convert images into Cartoon, Pencil Sketch or Edge Art</p>', unsafe_allow_html=True)

st.divider()

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg","jpeg","png"])

if uploaded_file:

    image = Image.open(uploaded_file)
    img = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼 Original Image")
        st.image(img, use_column_width=True)

    option = st.selectbox(
        "✨ Select Filter",
        ["Cartoon", "Pencil Sketch", "Edge Detection"]
    )

    result = None

    # Cartoon Filter
    if option == "Cartoon":

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray,5)

        edges = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            9,
            9
        )

        color = cv2.bilateralFilter(img,9,250,250)
        result = cv2.bitwise_and(color,color,mask=edges)

    # Pencil Sketch
    elif option == "Pencil Sketch":

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        invert = cv2.bitwise_not(gray)

        blur = cv2.GaussianBlur(invert,(21,21),0)
        inverted_blur = cv2.bitwise_not(blur)

        result = cv2.divide(gray,inverted_blur,scale=256.0)

    # Edge Detection
    elif option == "Edge Detection":

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = cv2.Canny(gray,100,200)

    with col2:

        st.subheader("🎯 Filtered Image")
        st.image(result, use_column_width=True)

        # Convert image for download
        img_pil = Image.fromarray(result)

        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")

        st.download_button(
            label="⬇ Download Image",
            data=buf.getvalue(),
            file_name="filtered_image.png",
            mime="image/png"
        )
