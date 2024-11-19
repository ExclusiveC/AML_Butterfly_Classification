import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Загрузите вашу модель
model = tf.keras.models.load_model('my_model.h5')

# Загрузите имена классов
class_names = {v: k for k, v in train_generator.class_indices.items()} #  train_generator из вашего ноутбука

# Создайте заголовок приложения
st.title('Классификация бабочек')

# Загрузите изображение
uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Преобразуйте изображение
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    # Сделайте прогноз
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    # Отобразите результат
    st.image(img, caption='Загруженное изображение', use_column_width=True)
    st.write(f"**Прогноз:** {predicted_class}")