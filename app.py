import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
# import pickle


import pickle

# Load the trained model
model_path = "D:\App_predictions\model_prediksi_gula2.pkl"
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Title of the app
st.title("Prediksi Kandungan Gula")

# Upload CSV data
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    data = pd.read_csv(uploaded_file)

    # # Display the exact column names to check for issues
    # st.write("Columns in the uploaded CSV file (with exact representation):")
    # st.write([repr(col) for col in data.columns])
    
    # # Strip any leading/trailing spaces
    # data.columns = data.columns.str.strip()

    # if 'Reducing Sugar' in data.columns:
    #     y_test = data['Reducing Sugar']
        
    #     # Drop the target column from the feature set
    #     X_test = data.drop(columns=['Reducing Sugar'])
        
    #     # Display the uploaded data
    #     st.write("Uploaded Data:")
    #     st.dataframe(data)
    
    # Display the uploaded data
    st.write("Uploaded Data:")
    st.dataframe(data)
    
    y_test = data['Reducing Sugar']
   
    data = data.drop(columns=['Reducing Sugar'])

    
    # Predict button
    if st.button("Predict"):
        # Make predictions
        predictions = model.predict(data)

         # Calculate metrics
        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mape = mean_absolute_percentage_error(y_test, predictions)
        
        # Create a new DataFrame to store the actual and predicted values
        result_df = pd.DataFrame({
            'Actual Reducing Sugar': y_test,
            'Predicted Reducing Sugar': predictions
        })
        
         # Create two columns for side-by-side display
        col1, col2 = st.columns([2, 1])  # Adjust the ratio if needed

        with col1:
                # Tabel Perbandingan
                st.write("**Perbandingan Gula Asli dan Gula Hasil Prediksi:**")
                st.dataframe(result_df)
            
        with col2:
                # Metrik Evaluasi
                st.write("**Metrics:**")
                st.write(f"R²: {r2:.4f}")
                st.write(f"RMSE: {rmse:.4f}")
                st.write(f"MAPE: {mape * 100:.2f}%")

        # # Display the comparison table
        # st.write("Perbandinga Gula Asli dan Gula Hasil Prediksi :")
        # st.dataframe(result_df)

      
            
        