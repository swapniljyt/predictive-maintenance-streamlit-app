import streamlit as st
import numpy as np
import pickle


model_filename = '/root/src/predictive-maintenance-streamlit-app/models/best_model.pkl'  # Update this path
with open(model_filename, 'rb') as file:
    model = pickle.load(file)


pca_filename = '/root/src/predictive-maintenance-streamlit-app/models/pca_transformer.pkl'  # Update this path
with open(pca_filename, 'rb') as file:
    pca = pickle.load(file)


st.title('Predictive Maintenance Model')


st.header('Enter the input features:')
air_temp = st.number_input('Air temperature [K]', value=298.1)
process_temp = st.number_input('Process temperature [K]', value=308.6)
rot_speed = st.number_input('Rotational speed [rpm]', value=1551)
torque = st.number_input('Torque [Nm]', value=42.8)
tool_wear = st.number_input('Tool wear [min]', value=0.0)


twf = st.selectbox('TWF (Tool Wear Failure)', [0, 1])
hdf = st.selectbox('HDF (Heat Dissipation Failure)', [0, 1])
pwf = st.selectbox('PWF (Power Failure)', [0, 1])
osf = st.selectbox('OSF (Overstrain Failure)', [0, 1])


if st.button('Predict'):

    input_data = np.array([[air_temp, process_temp, rot_speed, torque, tool_wear, twf, hdf, pwf, osf]])
    input_data_pca = pca.transform(input_data)
    prediction = model.predict(input_data_pca)
    st.write(f'The predicted output is: {prediction[0]}')

