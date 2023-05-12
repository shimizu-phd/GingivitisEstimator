import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd

labels = ["歯肉炎ではありません", "軽度の歯肉炎です", "中から重度の歯肉炎です"]
n_class = len(labels)
img_size = 224
n_result = 1

st.title('Gingivitis Estimator')
st.write('犬および猫の歯肉炎を判定します.')
st.write('左側のサイドバーような臼歯の周辺画像が必要です')
st.write('上下は不問です.')
st.write('')

st.sidebar.title('サンプル画像')
st.sidebar.write('保存して使用してください.')
st.sidebar.image('./images/c35-0.jpg', caption='猫-歯肉炎なし')
st.sidebar.image('./images/C30-2.jpg', caption='猫-中度歯肉炎')
st.sidebar.image('./images/c69-2.jpg', caption='猫-重度歯肉炎')
st.sidebar.image('./images/d70-1.jpg', caption='犬-歯肉炎なし')


@st.cache
def load_model():
    return tf.keras.models.load_model('./my_model_EN_adam.h5')


new_model = load_model()

uploaded_file = st.file_uploader("ファイルアップロード", type=['png', 'jpg', 'jpeg', 'webp'])

if uploaded_file is None:
    pass
else:
    image = Image.open(uploaded_file)
    image = image.convert("RGB")
    image_show = np.array(image)

    st.image(image_show, caption='uploaded image', use_column_width=True)

    image = image.resize((img_size, img_size))
    image = np.array(image)
    image = image.astype('float') / 255.0
    image = tf.reshape(image, [1, 224, 224, 3])

    pred = new_model.predict(image)
    sorted_idx = np.argsort(-pred[0])  # 降順でソート

    st.header('Result')

    for i in range(n_result):
        idx = sorted_idx[i]
        ratio = pred[0][idx]
        label = labels[idx]
        st.write(label)

    chart_data = pd.DataFrame(
        pred[0] * 100,
        index=labels,
        columns=['probability(%)'])
    st.bar_chart(chart_data)
