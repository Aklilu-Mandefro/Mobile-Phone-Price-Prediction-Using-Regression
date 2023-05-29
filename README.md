# Mobile Price Prediction Using Regression

This repository contains code for predicting the price of mobile phones using regression models. The project involves the application of various regression techniques, such as Random Forest Regression, Decision Tree Regressor, Linear Regression, and Lasso Regression, to estimate the price of mobile phones based on the given parameters. It also includes exploratory data analysis (EDA) and outlier analysis methods to gain insights into the dataset.

## Models and Accuracy

The implemented regression models have been evaluated using the dataset to estimate the price of mobile phones. The accuracy scores achieved by the models are as follows:

- Random Forest Regression: 96.94%
- Decision Tree Regressor: 96.79%
- Linear Regression: 96.14%
- Lasso Regression: 96.13%

Please note that these accuracy scores are specific to the given dataset and may vary depending on the data and problem statement.

## Streamlit Application

A Streamlit application has been developed to provide a user-friendly interface for interacting with the mobile price prediction model. The application allows users to input relevant parameters of a mobile phone and receive a predicted price based on the trained regression models. The application can be accessed through the following link: [Mobile Price Prediction Application](https://deepankarvarma-mobile-phone-price-prediction-using-r-app-xck2zi.streamlit.app/)

## Getting Started

To run the code in this repository locally, you can follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repository.git
```

2. Ensure that you have the necessary dependencies installed. The code requires the pandas library and scikit-learn for implementing the regression models. Install the dependencies using the following command:

```bash
pip install pandas scikit-learn
```

3. Import the necessary libraries and load the dataset:

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("mobile_data.csv")
```

4. Split the dataset into training and testing sets:

```python
X = data.drop("price", axis=1)  # Input features
y = data["price"]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

5. Implement and train the regression model, for example, Random Forest Regression:

```python
model = RandomForestRegressor()
model.fit(X_train, y_train)
```

6. Predict mobile phone prices using the trained model:

```python
y_pred = model.predict(X_test)
```

Feel free to explore the code and modify it according to your requirements.

## Contributing

Contributions to this repository are highly welcome. If you have any ideas, suggestions, or improvements, please feel free to fork the repository and submit a pull request. Let's collaborate to enhance the mobile price prediction model!

## License

This project is licensed under the [MIT License](LICENSE). You are free to use and modify the code for your own purposes.

