import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load mô hình hồi quy tuyến tính đã được lưu
model = pickle.load(open('linear.pkl', 'rb'))
# load mô hình chuẩn hóa Min-Max từ tệp "minmax_scaler_x.pkl"
with open("minmax_scaler_X.pkl", "rb") as scaler_file:
    loaded_minmax_scale = pickle.load(scaler_file)

# Define a function to make predictions
def predict_salary(BasePay, OvertimePay, OtherPay, Benefits, TotalPay, JobTitle):
    # Tạo mảng 1 chiều từ input_data
    input_data = np.array([BasePay, OvertimePay, OtherPay, Benefits, TotalPay, 0, 0, 0, 0, 0])
    if JobTitle == 'Police Captain':
        input_data [5] = 1
    elif JobTitle == 'Deputy Fire Chief':
        input_data [6] = 1
    elif JobTitle == 'Transit Mgr':
        input_data [7] = 1
    elif JobTitle == 'Cable Mechanic':
        input_data [8] = 1
    elif JobTitle == 'Other':
        input_data [9] = 1



    # Chuẩn hóa dữ liệu input_data bằng mô hình chuẩn hóa Min-Max
    input_data_normalized = loaded_minmax_scale.transform(input_data.reshape(1, -1))

    # Dự đoán giá trị bằng mô hình Linear Regressor đã nạp
    predicted_salary = model.predict(input_data_normalized)

    return predicted_salary[0]

# Create a Streamlit web app
st.title('San Francisco Employee Salary Prediction')
st.sidebar.header('Input Features')

# Input fields for user to enter feature values
BasePay = st.sidebar.number_input('Base Pay', value=0.0)
OvertimePay = st.sidebar.number_input('Overtime Pay', value=0.0)
OtherPay  = st.sidebar.number_input('Other Pay', value=0.0)
Benefits = st.sidebar.number_input('Benefits', value=0.0)
TotalPay = st.sidebar.number_input('Total Pay', value=0.0)
JobTitle = st.sidebar.selectbox('JobTitle', ('Police Captain', 'Deputy Fire Chief', 'Transit Mgr', 'Cable Mechanic', 'Other'))

# Định nghĩa CSS trực tiếp bằng cách sử dụng st.markdown
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f0f0; /* Nền màu xám nhạt */
    }
    .sidebar {
        background-color: #e0e0e0; /* Nền màu xám đậm hơn */
    }
    .title {
        color: #007bff; /* Màu xanh dương */
    }
    </style>
     <style>
    h1 {
        color: red;
        font-size: 36px;
    }
    </style>
    <style>
    .red-text {
        color: red;
        font-size: 30px;  /* Thay đổi cỡ chữ thành 30px */
    }
    </style>

    <style>
    .edit-text_blue {
    color: #007FFF; /* Xanh dương sáng */
    font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
#Tạo 1 list để lưu dữ liệu dự đoán sau mỗi lần ấn nút predict bằng session state
if "predicted_salaries" not in st.session_state:
    st.session_state.predicted_salaries = []

# Calculate the predicted
if st.sidebar.button('Predict'):
    predicted_salary = predict_salary(BasePay, OvertimePay, OtherPay, Benefits, TotalPay,JobTitle )
    predicted_salary = float(predicted_salary)
    st.session_state.predicted_salaries.append(predicted_salary)
    st.markdown(f'<p class="red-text">Predicted Salary: ${(predicted_salary):,.2f}</p>', unsafe_allow_html=True)

    # Vẽ biểu đồ dựa trên danh sách predicted_salaries với nền đen
    plt.figure(figsize=(8, 6), facecolor='black')
    plt.plot(range(1, len(st.session_state.predicted_salaries) + 1), st.session_state.predicted_salaries, color='lime')
    plt.xlabel('Prediction Number', color='white')
    plt.ylabel('Predicted Salary', color='white')
    plt.title('Biểu đồ  dự đoán lương nhân viên San Fransico', color='white')
    plt.gcf().set_facecolor('black')
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')
    st.pyplot(plt)
    # Đặt lại biểu đồ để tránh trùng lặp
    plt.clf()
    # Hiển thị danh sách giá dự đoán từ các lần nhấn trước đó
    if st.session_state.predicted_salaries:
        st.write("Danh sách lương dự đoán từ các lần nhấn trước đó:")
        for i, salary in enumerate(st.session_state.predicted_salaries):
            st.markdown(f'<p class="edit-text_blue">Dự đoán {i + 1}: ${salary:,.2f}</p>', unsafe_allow_html=True)

        # Hiển thị hình ảnh dựa trên giá trị predicted_price
        if predicted_salary < 100000:
            st.image("Image1 (1).png", use_column_width=True)
        elif 100000 <= predicted_salary <= 200000:
            st.image("Image2 (1).png", use_column_width=True)
        else:
            st.image("Image3 (1).png", use_column_width=True)

# This will clear the user inputs
if st.sidebar.button('Reset'):
    # Đặt lại tất cả các giá trị về giá trị mặc định
    st.session_state['BasePay'] = 0.0
    st.session_state['OvertimePay'] = 0.0
    st.session_state['OtherPay'] = 0.0
    st.session_state['Benefits'] = 0.0
    st.session_state['TotalPay'] = 0.0
    st.session_state['JobTitle'] = 'Police Captain'

    # Xóa danh sách predicted_prices
    st.session_state.predicted_salaries = []
# Provide some information about the app
st.write('This app predicts San Francisco employee salaries using a Linear Regression model.')

