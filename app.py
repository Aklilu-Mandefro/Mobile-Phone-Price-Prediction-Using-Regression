import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the dataset into a pandas dataframe
df = pd.read_csv('data.csv')

df.drop('Unnamed: 0',axis=1,inplace=True)
# Check for missing values
print(df.isnull().sum())

# Replace missing values with the mean
df.fillna(df.mean(), inplace=True)

# # Check for outliers
# # You can use box plots or scatter plots to identify outliers
# # Remove outliers that are more than 3 standard deviations away from the mean
df = df[(df['Price'] - df['Price'].mean()).abs() < 3 * df['Price'].std()]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop({'Price','Brand me'}, axis=1), df['Price'], test_size=0.2, random_state=42)

# Train a random forest regression model on the training set
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create a Streamlit app
st.set_page_config(page_title='Cellphone Price Predictor')
st.title("Cellphone Price Predictor")
st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://www.10wallpaper.com/wallpaper/1920x1200/1908/2019_Purple_Abstract_4K_HD_Design_1920x1200.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
# Create a form for the user to input features
form = st.form(key='input_form')
ratings = form.number_input("Ratings", min_value=0.0, max_value=5.0, step=0.1)
ram = form.number_input("RAM")
rom = form.number_input("ROM")
mobile_size = form.number_input("Mobile Size")
primary_cam = form.number_input("Primary Camera")
selfi_cam = form.number_input("Selfie Camera")
battery_power = form.number_input("Battery Power")
submit_button = form.form_submit_button(label='Submit')

# Make a prediction based on the user's input
if submit_button:
    input_data = [[ratings, ram, rom, mobile_size, primary_cam, selfi_cam, battery_power]]
    prediction = model.predict(input_data)[0]
    st.header(f"The predicted price is {prediction:.2f}")
