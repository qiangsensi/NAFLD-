import streamlit as st
import pickle
import pandas as pd
import numpy as np
import shap
from sklearn.preprocessing import StandardScaler
import joblib

#显示中文
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 加载保存的模型
@st.cache_resource
def load_model():
    with open('rf.pkl', 'rb') as f:
        model = joblib.load(f)
    return model

# 加载数据（假设你有一个用于特征的标准数据集）
@st.cache_data
def load_data():
    data = pd.read_excel('总数据.xlsx')
    features = ["年龄","BMI", "谷丙转氨酶", 
                   "白蛋白", 
                   "TyG指数",
                    "糖化血红蛋白", 
                   "PHR", "SUA/CR"
                ]
    data = data[features]
    return data

# 主函数
def main():
    st.title("基于随机森林算法的NAFLD临床诊断预测模型")
    # st.write("这是一个基于随机森林模型解释和预测应用。")

    # 加载模型和数据
    model = load_model()
    data = load_data()

    # 显示数据
    st.subheader("数据集")
    st.write(data.head())

    # 在侧边栏中用户输入特征值
    st.sidebar.subheader("输入特征值")
    features = {}
    for col in data.columns:
        features[col] = st.sidebar.number_input(
            f"输入 {col}", 
            value=float(data[col].mean())  # 默认值为该特征的均值
        )

    # 将输入转换为 DataFrame
    input_data = pd.DataFrame([features])

    # 标准化输入数据（如果模型需要）
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_input = scaler.transform(input_data)

    # 预测
    if st.sidebar.button("预测"):
        prediction = model.predict(scaled_input)
        st.success(f"预测结果: {prediction}")

        # SHAP 解释
        explainer = shap.Explainer(model, scaled_data)
        shap_values = explainer(scaled_input)

        # 检查 shap_values 的形状
        # st.write(f"SHAP 值的形状: {shap_values.shape}")

        #提取单个样本的shape值与期望值
        shap_values_single = shap_values[0]
        expected_value = explainer.expected_value[0]

        #创建explanation对象
        explanation = shap.Explanation(
            values=shap_values_single[:,0],
            base_values=expected_value,
            data=input_data.iloc[0].values,
            feature_names=data.columns.tolist() 
        )

        shap.save_html("shap.html", shap.plots.force(explanation,show = False))

        st.subheader("SHAP 值")
        with open("shap.html", 'r', encoding='utf-8') as f:
            st.components.v1.html(f.read(), height=800)

if __name__ == "__main__":
    main()