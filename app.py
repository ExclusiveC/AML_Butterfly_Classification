import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

model = tf.keras.models.load_model('my_model.h5')

class_names = {0: 'ADONIS', 1: 'AFRICAN GIANT SWALLOWTAIL', 2: 'AMERICAN SNOOT', 3: 'AN 88', 4: 'APPOLLO', 5: 'ATALA', 6: 'BANDED ORANGE HELICONIAN', 7: 'BECKERS WHITE', 8: 'BLACK HAIRSTREAK', 9: 'BROWN SIPROETA', 10: 'CABBAGE WHITE', 11: 'CAIRNS BIRDWING', 12: 'CHECQUERED SKIPPER', 13: 'CLEOPATRA', 14: 'CLOUDED SULPHUR', 15: 'COMMON BANDED AWL', 16: 'COMMON WOOD-NYMPH', 17: 'COPPER TAIL', 18: 'CRECENT', 19: 'CRIMSON PATCH', 20: 'DANAID EGGFLY', 21: 'EASTERN COMMA', 22: 'EASTERN PINE ELFIN', 23: 'ELBOWED PIERROT', 24: 'GOLD BANDED', 25: 'GREAT EGGFLY', 26: 'GREAT JAY', 27: 'GREEN CELLED CATTLEHEART', 28: 'GREY HAIRSTREAK', 29: 'INDRA SWALLOW', 30: 'IPHICLUS SISTER', 31: 'JULIA', 32: 'LARGE MARBLE', 33: 'MALACHITE', 34: 'MANGROVE SKIPPER', 35: 'MESTRA', 36: 'METALMARK', 37: 'MILBERTS TORTOISESHELL', 38: 'MONARCH', 39: 'MOURNING CLOAK', 40: 'ORANGE SKIPPERLING', 41: 'ORCHARD SWALLOW', 42: 'PAINTED LADY', 43: 'PAPER KITE', 44: 'PEACOCK', 45: 'PIPEVINE SWALLOW', 46: 'POPINJAY', 47: 'PURPLE HAIRSTREAK', 48: 'QUESTION MARK', 49: 'RED ADMIRAL', 50: 'RED CRACKER', 51: 'RED POSTMAN', 52: 'RED SPOTTED PURPLE', 53: 'SCARCE SWALLOW', 54: 'SILVER SPOT SKIPPER', 55: 'SLEEPY ORANGE', 56: 'SOUTHERN DOGFACE', 57: 'STRAITED QUEEN', 58: 'TROPICAL LEAFWING', 59: 'TWO BARRED FLASHER', 60: 'ULYSES', 61: 'VICEROY', 62: 'WHITE ADMIRAL', 63: 'WOOD SATYR', 64: 'YELLOW SWALLOW TAIL', 65: 'ZEBRA LONGWING'}

def preprocess_image(image_file):
    img = Image.open(image_file)
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 
    return img_array

st.title("Классификация бабочек")
uploaded_file = st.file_uploader("Загрузите изображение бабочки", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    image = preprocess_image(uploaded_file)

    # Предсказание
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class]

    # Отображение результатов
    st.image(image, caption='Загруженное изображение', use_column_width=True)
    st.write(f"**Предсказанный класс:** {predicted_class_name}")
