## INSY 662 GROUP PROJECT - PREDICTING OTTAWA HOUSE PRICES

# Step 0 : Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import iqr
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset
data = pd.read_csv("C:/Users/aadel/Downloads/ottawa-realestate-data.csv")

# Remove commas from 'price' column and convert to numeric
data['price'] = pd.to_numeric(data['price'].str.replace(',', ''))


# Step 2: Inspect the first few rows
print(data.head())
data.describe()


# Step 3: Check for missing values
print(data.isnull().sum())


# Step 4: Handle missing values
# Impute missing values with mean for the following columns
columns_to_fill_with_mean = ['lotDepth', 'lotFrontage', 'garage', 'parking', "latitude", "longitude"]
for col in columns_to_fill_with_mean:
    data[col].fillna(data[col].mean(), inplace=True)

# Drop rows with missing values in the following columns
columns_to_drop_rows = ['yearBuilt', 'style', 'bedrooms', 'bathrooms', 'walkScore', 'price', "propertyType"]
for col in columns_to_drop_rows:
    data.dropna(subset=[col], inplace=True)

# Check for missing values again
print(data.isnull().sum())


# Step 5: Add two new columns
# Calculate the distance from downtown using the latitude and longitude
downtown_lat = 45.42519984540492
downtown_long = -75.69992181099843
data['distanceFromDowntown'] = np.sqrt((data['latitude'] - downtown_lat)**2 + (data['longitude'] - downtown_long)**2)

# Calculate lotSize using lotDepth and lotFrontage
data['lotSize'] = data['lotDepth'] * data['lotFrontage']


# Step 6: Remove lat/longitude columns since distanceFromDowntown is added
data.drop(columns=['longitude', 'latitude'], inplace=True)


# Step 7: Exclude the 'postalCode' column
data = data.drop(columns=['postalCode'])


# Step 8: Filter only residential properties
data = data[data['propertyType'] == 'Residential']


# Step 9: Dummify categorical variables
data_encoded = pd.get_dummies(data, columns=['propertyType', 'style'], drop_first=True)


# Step 10: Check for multicollinearity using VIF
# Filter out only numeric data
numeric_data = data_encoded.select_dtypes(include=[np.number])
# Remove the 'price' column as we don't calculate VIF for the dependent variable
numeric_data = numeric_data.drop(columns=['price'])
# Now, you can calculate VIF
vif_data = pd.DataFrame()
vif_data["feature"] = numeric_data.columns
vif_data["VIF"] = [variance_inflation_factor(numeric_data.values, i) for i in range(numeric_data.shape[1])]

print(vif_data)

# Generally, a VIF above 5-10 indicates a problematic amount of collinearity.
# You can then decide which features to keep or remove based on their VIF values.
high_vif_features = vif_data[vif_data["VIF"] > 10]["feature"].tolist()
print(f"Features with high VIF (indicative of multicollinearity): {high_vif_features}")

#CORRELATION HEATMAP
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Calculate the correlation matrix
correlation_matrix = data_encoded.corr()

# Create a heatmap using Seaborn
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')

plt.title('Correlation Heatmap', pad=20)
plt.show()


# Step 11: Remove the yearBuilt column due to multicollinearity
data_encoded.drop(columns=['yearBuilt'], inplace=True)



#Step 11.1: Create Visualization graphs for EDA

# Scatter plot: Bedrooms vs Price
plt.figure(figsize=(8, 6))
plt.scatter(data_encoded['bedrooms'], data_encoded['price'])
plt.title('Bedrooms vs Price')
plt.xlabel('price')
plt.ylabel('bedrooms')
plt.grid(True)
plt.show()

# Scatter plot: Bathrooms vs Price
plt.figure(figsize=(8, 6))
plt.scatter(data_encoded['bathrooms'], data_encoded['price'])
plt.title('Bathrooms vs Price')
plt.xlabel('price')
plt.ylabel('bathrooms')
plt.grid(True)
plt.show()

#SCATTER PLOT THAT CONSIDERS VARIABLES THAT ARE RELATED TO THE ROOMS 

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Define colors for each variable
colors = ['orange', 'red', 'green', 'blue']

# Scatter plot: Multiple variables on x-axis vs Price on y-axis with different colored dots
plt.figure(figsize=(8, 6))

for i, var in enumerate(['bedrooms', 'bathrooms','parking', 'garage']):
    # Modify the label names for the legend
    if var == 'bedrooms':
        label_name = 'Bedrooms'
    elif var =='bathrooms':
        label_name = 'Bathrooms'
    elif var =='parking':
        label_name = 'Parking'
    else:
        label_name = 'Garage'
    plt.scatter(data_encoded[var], data_encoded['price'], color=colors[i], label=label_name)

# Add labels and title
plt.title('Rooms vs Price')
plt.xlabel('Variables')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

# Set the formatter for the y-axis ticks to display prices without scientific notation
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{y:,.0f}'))

plt.show()

#Histogram

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Histogram of prices
plt.figure(figsize=(8, 6))
plt.hist(data['price'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')

# Set the formatter for the x-axis ticks
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:,.0f}'))

plt.grid(True)
plt.show()

#Barplot for Bedrooms

# Define a single color for all bars
single_color = 'orange'

# Bar plot: Bedrooms vs Price with a single color for all bars
plt.figure(figsize=(8, 6))
bars = plt.bar(data['bedrooms'], data['price'], color=single_color)

# Add labels and title
plt.title('Price of House by Number of Bedrooms')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Price')
plt.xticks(data['bedrooms'])  # Set x-axis ticks to bedrooms values

# Set the formatter for the x-axis ticks
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{y:,.0f}'))

plt.grid(False)
plt.show()

#Barplot for Bathrooms

# Define a single color for all bars
single_color = 'orange'

# Bar plot: Bathrooms vs Price with a single color for all bars
plt.figure(figsize=(8, 6))
bars = plt.bar(data['bathrooms'], data['price'], color=single_color)

# Add labels and title
plt.title('Price of House by Number of Bathrooms')
plt.xlabel('Number of Bathrooms')
plt.ylabel('Price')
plt.xticks(data['bathrooms'])  # Set x-axis ticks to bedrooms values

# Set the formatter for the x-axis ticks
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{y:,.0f}'))

plt.grid(False)
plt.show()



# Step 12: Split data into train and test sets
X = data_encoded.drop(columns=['price'])
y = data_encoded['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 13: Train multiple models
# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin_reg = lin_reg.predict(X_test)
mse_lin_reg = mean_squared_error(y_test, y_pred_lin_reg)
r2_lin_reg = r2_score(y_test, y_pred_lin_reg)

# Random Forest Regressor
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Gradient Boosting Regressor
gb_reg = GradientBoostingRegressor(random_state=42)
gb_reg.fit(X_train, y_train)
y_pred_gb = gb_reg.predict(X_test)
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)


# Step 14: Evaluate models using MSE + R squared
print(f'MSE for Linear Regression: {mse_lin_reg:.4e}')
print(f'MSE for Random Forest: {mse_rf:.4e}')
print(f'MSE for Gradient Boosting: {mse_gb:.4e}')

print(f'R-squared for Linear Regression: {r2_lin_reg:.4}')
print(f'R-squared for Random Forest: {r2_rf:.4}')
print(f'R-squared for Gradient Boosting: {r2_gb:.4}')


# Step 15: Hyperparameter tuning for Gradient Boosting Regressor
# Define ranges for hyperparameters
n_estimators_list = [50, 100, 200]
learning_rate_list = [0.01, 0.05, 0.1]
max_depth_list = [3, 4, 5]

best_mse = float('inf')  # Initialize with a very high MSE value
best_params = {}

# Hyperparameter tuning using nested loops
for n_estimators in n_estimators_list:
    for learning_rate in learning_rate_list:
        for max_depth in max_depth_list:
            # Train the Gradient Boosting model with current hyperparameters
            gb_model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42)
            gb_model.fit(X_train, y_train)
            
            # Evaluate the model performance
            y_pred_gb = gb_model.predict(X_test)
            mse_gb = mean_squared_error(y_test, y_pred_gb)
            
            # Update best parameters if this configuration improves MSE
            if mse_gb < best_mse:
                best_mse = mse_gb
                best_params = {
                    'n_estimators': n_estimators,
                    'learning_rate': learning_rate,
                    'max_depth': max_depth
                }


#Step 16:  Retrain the Gradient Boosting model with the best hyperparameters
best_gb_model = GradientBoostingRegressor(
    n_estimators=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    max_depth=best_params['max_depth'],
    random_state=42)
best_gb_model.fit(X_train, y_train)

# Evaluate the model performance
y_pred_best_gb = best_gb_model.predict(X_test)
mse_best_gb = mean_squared_error(y_test, y_pred_best_gb)
r2_best_gb = r2_score(y_test, y_pred_best_gb)

print(f'Best Hyperparameters: {best_params}')
print(f'MSE for Best Gradient Boosting Model: {mse_best_gb:.4e}')
print(f'R-squared for Best Gradient Boosting: {r2_best_gb:.4}')


# Step 17: Compare models and choose the best one
# We tested Linear Regression, Random Forest, and Gradient Boosting and chose Gradient Boosting based on its performance.


# Step 18: Train the chosen model (Gradient Boosting) on the entire dataset
final_model = GradientBoostingRegressor(
    n_estimators=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    max_depth=best_params['max_depth'],
    random_state=42)
final_model.fit(X, y)


# Step 19 : Make predictions

# 1. Predictions for the test set
# Using the Gradient Boosting Model trained on the training set
y_pred_gb = best_gb_model.predict(X_test)

# 2. Predictions for Audrey and Dhevin's ideal home scenario
# Define the features for Audrey and Dhevin's ideal home
ideal_home = {
    'bedrooms': [4],
    'bathrooms': [2],
    'garage': [2.0],
    'parking': [2.0],
    'walkscore': [60],
    'yearBuilt': [2005], # Assuming a year midway through the 2000s
    'propertyType_Residential': [1],
    'style_2 Storey': [1], # Assuming a common style for clarity; can be adjusted
    'distanceFromDowntown': [0.015], # Rough estimate for a 20-minute commute; can be adjusted
    'lotSize': [4000] # Rough estimate; can be adjusted
}

# Create a dataframe with the same columns as X_train filled with zeros
ideal_home_df = pd.DataFrame(np.zeros((1, X_train.shape[1])), columns=X_train.columns)

# Update the values in ideal_home_df using the ideal_home dictionary
for key, value in ideal_home.items():
    if key in ideal_home_df.columns:
        ideal_home_df[key] = value

# Now, predict using the best_gb_model
predicted_price_ideal_home = best_gb_model.predict(ideal_home_df)
print(f'Predicted Price for Ideal Home: {predicted_price_ideal_home}')

# 3. Predictions for a set of fake data
# Define a set of hypothetical properties (fake data for prediction)
fake_data = {
    'bedrooms': [3, 4, 2, 3, 3, 4, 5, 3, 2, 4],
    'bathrooms': [2, 3, 1, 2, 2, 3, 4, 2, 1, 3],
    'garage': [1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0],
    'parking': [1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 1.0, 2.0],
    'walkscore': [50, 60, 55, 52, 58, 65, 70, 54, 53, 63],
    'yearBuilt': [2000, 2010, 1995, 2005, 2005, 2015, 2018, 2002, 1998, 2012],
    'propertyType_Residential': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'style_2 Storey': [1, 0, 1, 1, 0, 0, 0, 1, 1, 0],
    'distanceFromDowntown': [0.02, 0.01, 0.03, 0.02, 0.015, 0.01, 0.009, 0.02, 0.025, 0.015],
    'lotSize': [3500, 4000, 3000, 3700, 3800, 4100, 4200, 3600, 3200, 4000]
}

# Create a dataframe with the same columns as X_train filled with zeros for each of the 10 hypothetical properties
fake_data_df = pd.DataFrame(np.zeros((10, X_train.shape[1])), columns=X_train.columns)

# Update the values in fake_data_df using the fake_data dictionary
for key, value in fake_data.items():
    if key in fake_data_df.columns:
        fake_data_df[key] = value

# Now, predict using the best_gb_model
predicted_prices_fake_data = best_gb_model.predict(fake_data_df)


# Step 20 : Interpreting Results (graphs)

# Define the features for Audrey and Dhevin's ideal home (as previously defined)
ideal_home = {
    'Feature': ['Bedrooms', 'Bathrooms', 'Garage', 'Parking', 'Walkscore', 
                'Year Built', 'Property Type (Residential)', 'Style (2 Storey)',
                'Distance from Downtown', 'Lot Size', 'Predicted Price'],
    'Value': [4, 2, 2.0, 2.0, 60, 2005, 1, 1, 0.015, 4000, predicted_price_ideal_home[0]]
}

# Convert to DataFrame
ideal_home_df = pd.DataFrame(ideal_home)

# Create a table plot
fig, ax = plt.subplots(figsize=(6, 3))  # Adjust figure size as needed
ax.axis('tight')
ax.axis('off')
ax.table(cellText=ideal_home_df.values, colLabels=ideal_home_df.columns, loc='center')

plt.title('Ideal Home Features and Predicted Price', y=1)  # Adjust title position
plt.show()


# Generate the Distribution of Actual vs. Predicted Prices for the Gradient Boosting Model's predictions
plt.figure(figsize=(10, 6))
plt.hist(y_test, bins=30, alpha=0.5, color='green', label='Actual Prices')
plt.hist(y_pred_gb, bins=30, alpha=0.5, color='blue', label='Predicted Prices')
plt.xlabel('Price', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of Actual vs. Predicted Prices', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()


#Plot graph for Ideal Home Output
import matplotlib.pyplot as plt
import pandas as pd

# Define the features for Audrey and Dhevin's ideal home (as previously defined)
ideal_home = {
    'Feature': ['Bedrooms', 'Bathrooms', 'Garage', 'Parking', 'Walkscore', 
                'Year Built', 'Property Type (Residential)', 'Style (2 Storey)',
                'Distance from Downtown', 'Lot Size', 'Predicted Price'],
    'Value': [4, 2, 2.0, 2.0, 60, 2005, 1, 1, 0.015, 4000, predicted_price_ideal_home[0]]
}

# Convert to DataFrame
ideal_home_df = pd.DataFrame(ideal_home)

# Create a table plot
fig, ax = plt.subplots(figsize=(6, 3))  # Adjust figure size as needed
ax.axis('tight')
ax.axis('off')
ax.table(cellText=ideal_home_df.values, colLabels=ideal_home_df.columns, loc='center')

plt.title('Ideal Home Features and Predicted Price', y=1)  # Adjust title position
plt.show()

# Scatter plot of actual vs predicted prices for the test set
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_gb, alpha=0.5)
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Diagonal line for reference
plt.show()

# Generate the Residual Plot for the Gradient Boosting Model's predictions
# Calculate residuals
residuals = y_test - y_pred_gb

# Plotting the residuals
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_gb, residuals, alpha=0.5, color='blue')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Prices', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.title('Residual Plot', fontsize=16)
plt.grid(True)
plt.show()

import seaborn as sns

# Distribution plot of predicted prices
plt.figure(figsize=(10, 6))
sns.histplot(y_pred_gb, kde=True, color='red')
plt.title('Distribution of Predicted Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


# Visualization for fake data predictions
plt.figure(figsize=(10, 6))
plt.bar(range(1, 11), predicted_prices_fake_data, color='lightgreen')
plt.title('Predicted Prices for Hypothetical Properties')
plt.xlabel('Property Index')
plt.ylabel('Price')
plt.xticks(range(1, 11))  # Set x-tick labels to property index
plt.show()



#Step 21 : Create Visualization Maps for EDA

import folium
import pandas as pd

# Load your dataset
data2 = pd.read_csv("C:/Users/aadel/Downloads/ottawa-realestate-data.csv")


# Remove commas from 'price' column and convert to numeric
data2['price'] = pd.to_numeric(data2['price'].str.replace(',', ''))

# Impute missing values with mean for the following columns
columns_to_fill_with_mean = ['lotDepth', 'lotFrontage', 'garage', 'parking', "latitude", "longitude"]
for col in columns_to_fill_with_mean:
    data2[col].fillna(data2[col].mean(), inplace=True)

# Drop rows with missing values in the following columns
columns_to_drop_rows = ['yearBuilt', 'style', 'bedrooms', 'bathrooms', 'walkScore', 'price', "propertyType"]
for col in columns_to_drop_rows:
    data2.dropna(subset=[col], inplace=True)


# Filter only residential properties
data2 = data2[data2['propertyType'] == 'Residential']


#Price house map

# Function to determine the color based on price
def price_color(price):
    if price < 300000:
        return 'green'
    elif 300000 <= price < 600000:
        return 'orange'
    else:
        return 'red'

# Create a map centered around an average location in Ottawa
map_ottawa = folium.Map(location=[45.4215, -75.6972], zoom_start=12)

# Add a marker for each property
for _, row in data2.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5, # size of markers
        color=price_color(row['price']),
        fill=True,
        fill_color=price_color(row['price']),
        fill_opacity=0.7,
        popup=f"Price: ${row['price']}",
    ).add_to(map_ottawa)

# Save the map to an HTML file
map_ottawa.save("C:/Users/aadel/Downloads/ottawa_price_distribution_map.html")
                        

##Price by Area Map


import pandas as pd
import geopandas as gpd
import folium
import json

# Assuming 'latitude' and 'longitude' are in your dataset and you want to use them to join with the GeoJSON file
gdf = gpd.GeoDataFrame(
    data2, geometry=gpd.points_from_xy(data2.longitude, data2.latitude))

# Load GeoJSON file
with open("C:/Users/aadel/Downloads/ons.geojson", 'r') as f:
    geojson_data = json.load(f)

# Convert GeoJSON to a GeoDataFrame
geojson_gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])

# Perform a spatial join between the real estate data and the GeoJSON areas
merged_data = gpd.sjoin(gdf, geojson_gdf, how="inner", op="within")

# Calculate average price for each area
average_price = merged_data.groupby('names').price.mean().reset_index()
average_price.columns = ['area', 'average_price']

# Create a folium map
map_ottawa = folium.Map(location=[45.4215, -75.6972], zoom_start=12)

# Create choropleth map
folium.Choropleth(
    geo_data=geojson_data,
    name='choropleth',
    data=average_price,
    columns=['area', 'average_price'],
    key_on='feature.properties.Neighborhood', # replace with correct property
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Average Property Price'
).add_to(map_ottawa)

# Save to HTML
map_ottawa.save("C:/Users/aadel/Downloads/ottawa_average_price_map.html")

                           