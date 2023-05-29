import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# Load the dataset into a pandas dataframe
df = pd.read_csv('data.csv')

df.drop('Unnamed: 0',axis=1,inplace=True)
# Check for missing values
print(df.isnull().sum())
print(df)
# Replace missing values with the mean
df.fillna(df.mean(), inplace=True)

# # Check for outliers
# # You can use box plots or scatter plots to identify outliers
# # Remove outliers that are more than 3 standard deviations away from the mean
df = df[(df['Price'] - df['Price'].mean()).abs() < 3 * df['Price'].std()]

# # Check for correlations
corr_matrix = df.corr()
print(corr_matrix['Price'].sort_values(ascending=False))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop({'Price','Brand me'}, axis=1), df['Price'], test_size=0.2, random_state=42)

# Train a random forest regression model on the training set
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the testing set and print the R2 score
r2 = model.score(X_test, y_test)
print("R2 score on testing set: ", r2)
