import streamlit as st
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def get_palette_kmeans(image, num_colors):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)
    img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
    img = img.reshape((img.shape[0] * img.shape[1], 3))

    kmeans = KMeans(n_clusters=num_colors, max_iter=300, n_init=10, random_state=42)
    kmeans.fit(img)

    centers = kmeans.cluster_centers_
    centers = np.array(centers, dtype='uint8')
    centers = cv2.cvtColor(np.array([centers]), cv2.COLOR_LAB2RGB)[0]
    return centers

def calculate_brightness(color):
    return np.sqrt(0.299 * (color[0] ** 2) + 0.587 * (color[1] ** 2) + 0.114 * (color[2] ** 2))

def display_palette(colors):
    sorted_colors = sorted(colors, key=calculate_brightness)
    
    color_blocks = []
    for color in sorted_colors:
        block = np.full((100, 100, 3), color, dtype=np.uint8)
        color_blocks.append(block)

    palette = np.hstack(color_blocks)
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.imshow(palette)
    ax.axis('off')
    st.pyplot(fig)

st.set_page_config(layout="centered", page_title="Color Picker Website")
st.title("Color Picker dari Gambar")
st.subheader("Upload gambar di sidebar")

uploaded_file = st.sidebar.file_uploader("Pilih gambar", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.write("Gambar yang diupload")
    image = Image.open(uploaded_file)
    st.image(image, width=700)
    num_colors = st.slider("Jumlah warna dominan", min_value=5, max_value=8, value=5)
    colors = get_palette_kmeans(image, num_colors)
    st.write("Palet warna dominan:")
    display_palette(colors)

else:
    st.write("Contoh gambar")
    image = Image.open("./dummy.jpg")
    st.image(image, width=700)
    num_colors = st.slider("Jumlah warna dominan", min_value=5, max_value=8, value=5)
    colors = get_palette_kmeans(image, num_colors)
    st.write("Palet warna dominan")
    display_palette(colors)
