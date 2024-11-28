import streamlit as st
from PIL import Image
from pycaret.regression import load_model
from sklearn.pipeline import Pipeline
import pandas as pd
# Page configuration
st.set_page_config(
    page_title='Prediction of bond strength and failure modes of CFRP-steel joints',
    layout='wide',
    initial_sidebar_state='expanded'
)


# Title of the app centered with HTML and CSS
st.markdown("""
    <style>
    .center-content {
        text-align: center;
    }
    </style>
    <h1 style='color: darkblue;' class='center-content'>Synthetic data-augmented and automated deep learning assessment of creep life in adhesively bonded joints</h1>
    """, unsafe_allow_html=True)

st.write(' ')
# Additional info centered
st.markdown("""
    <div class='center-content'>
    Developed by zhen liu,Hubei University of Technology.<br>
    Email: lz18246444022@163.com
    </div>
    """, unsafe_allow_html=True)

st.write(' ')
# Open and resize the image
image = Image.open("Figure1.jpg")
resized_image = image.resize((430, 200))
# Display the image centered
st.image(resized_image, use_column_width=False)

# Add a text input box
user_input = st.text_input("Please enter the relevant content here.", "")

st.write(' ')

tuned_best_regression = load_model('tuned_best_two_regression')

def extract_features_from_text(text):
    features_dict = {}
    parts = text.split(',')
    for part in parts:
        key_value = part.split(':')
        if len(key_value) == 2:
            key = key_value[0].strip()
            value = float(key_value[1].strip())
            features_dict[key] = value
    return features_dict
# 构建完整的管道，只包含模型预测这一个步骤
pipeline = Pipeline([
    ('regressor', tuned_best_regression)
])
# 示例输入的一段文本，包含多个特征相关信息
input_text =st.text_input( " ")
if st.button("Prediction Results"):
  if input_text:
    # 先从文本中提取特征并转换为字典
    extracted_features = extract_features_from_text(input_text)

    # 将提取的特征转换为DataFrame格式，这是管道输入数据通常期望的格式，这里的 orient='index' 表示以字典的索引（即特征名）作为列名，转置后符合模型输入要求
    input_data = pd.DataFrame.from_dict(extracted_features, orient='index').T

    # 使用管道进行预测
    prediction = pipeline.predict(input_data)
    st.write("Creep failure life:", prediction[0])
  else:
    st.write("请输入有效文本内容进行预测")