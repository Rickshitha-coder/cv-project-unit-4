import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(
    page_title="AI Photo Editor",
    page_icon="🎨",
    layout="wide"
)

# Header
st.title("🎨 AI Photo Editor")
st.write("Upload an image and apply professional filters.")

# Sidebar
st.sidebar.header("🛠 Editing Tools")

uploaded_file = st.sidebar.file_uploader(
    "Upload Image", type=["jpg","jpeg","png"]
)

filter_option = st.sidebar.selectbox(
    "Choose Filter",
    ["None","Cartoon","Pencil Sketch","Edge Detection"]
)

brightness = st.sidebar.slider("Brightness", -50, 50, 0)
contrast = st.sidebar.slider("Contrast", 0.5, 3.0, 1.0)

if uploaded_file:

    image = Image.open(uploaded_file)
    img = np.array(image)

    # Brightness & contrast adjustment
    img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

    result = img.copy()

    # Cartoon Filter
    if filter_option == "Cartoon":

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray,5)

        edges = cv2.adaptiveThreshold(
            gray,255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            9,9
        )

        color = cv2.bilateralFilter(img,9,250,250)

        result = cv2.bitwise_and(color,color,mask=edges)

    # Pencil Sketch
    elif filter_option == "Pencil Sketch":

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        invert = cv2.bitwise_not(gray)

        blur = cv2.GaussianBlur(invert,(21,21),0)

        inverted_blur = cv2.bitwise_not(blur)

        result = cv2.divide(gray,inverted_blur,scale=256.0)

    # Edge Detection
    elif filter_option == "Edge Detection":

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        result = cv2.Canny(gray,100,200)

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Edited Image")
        st.image(result, use_column_width=True)

    # Download image
    img_pil = Image.fromarray(result)

    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")

    st.download_button(
        "⬇ Download Edited Image",
        buf.getvalue(),
        file_name="edited_image.png",
        mime="image/png"
    )
