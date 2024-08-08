import streamlit as st
import numpy as np
import pickle
from sklearn.decomposition import PCA

# Load the trained model
model_filename = '/kaggle/working/best_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Define the Streamlit app
st.title('Predictive Maintenance Model')

# Input features
st.header('Enter the input features:')
air_temp = st.number_input('Air temperature [K]', min_value=0.0)
process_temp = st.number_input('Process temperature [K]', min_value=0.0)
rot_speed = st.number_input('Rotational speed [rpm]', min_value=0.0)
torque = st.number_input('Torque [Nm]', min_value=0.0)
tool_wear = st.number_input('Tool wear [min]', min_value=0.0)

# One-hot encoded categorical features
twf = st.selectbox('TWF (Tool Wear Failure)', [0, 1])
hdf = st.selectbox('HDF (Heat Dissipation Failure)', [0, 1])
pwf = st.selectbox('PWF (Power Failure)', [0, 1])
osf = st.selectbox('OSF (Overstrain Failure)', [0, 1])

# Create a button to make the prediction
if st.button('Predict'):
    # Prepare the input data
    input_data = np.array([[air_temp, process_temp, rot_speed, torque, tool_wear, twf, hdf, pwf, osf]])
    
    # Apply PCA transformation if your model expects PCA-transformed data
    pca2 = PCA(n_components=2)
    input_data_pca = pca2.fit_transform(input_data)
    
    # Make the prediction
    prediction = model.predict(input_data_pca)
    
    # Display the prediction
    st.write(f'The predicted output is: {prediction[0]}')

# To run the Streamlit app, save this script as `app.py` and run `streamlit run app.py` in your terminal