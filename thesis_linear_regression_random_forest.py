import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


## VARIABLES
num_features = 31 
target_variable = 'Earnings Per Share - Actual'



# Read the Excel file
quarterly_data = pd.read_excel('DATA_MICROSOFT_financial.xlsx')
print(quarterly_data)
yearly_data = pd.read_excel('DATA_MICROSOFT_esg.xlsx')
print(yearly_data)



# Create pandas dataframes from the data
df_year = pd.DataFrame(yearly_data)
df_quarter = pd.DataFrame(quarterly_data)

# # # Reverse the order of rows
df_year_reversed = df_year[::-1].reset_index(drop=True)
df_quarter_reversed = df_quarter[::-1].reset_index(drop=True)

df_year = df_year_reversed
df_quarter = df_quarter_reversed 





# ######################################################################################################################
# ## DATA CLEANING 


########################################################################################################################
## QUARTELY TO DATETIME -- the financial indicators


# Define a function to convert 'FQ-X' format to datetime
def convert_fq_to_datetime(fq_string):
    try:
        quarter = int(fq_string.split('-')[1])  # Extract the quarter number from the 'FQ-X' string
    except IndexError:
        return pd.NaT  # Return NaT if the split operation fails
    year = 2023 - quarter // 4  # Determine the year based on the quarter number
    month = (quarter % 4) * 3 + 3  # Determine the starting month of the quarter
    return pd.Timestamp(year=year, month=month, day=30)  # Assuming FY starts in January

# Convert 'timestamp' column to datetime format
df_quarter['timestamp'] = df_quarter['timestamp'].apply(convert_fq_to_datetime)

# Drop rows with NaT values (if any)
df_quarter.dropna(subset=['timestamp'], inplace=True)

# Set the datetime column as index
df_quarter.set_index('timestamp', inplace=True)

# print(df_quarter)

df_quarter = df_quarter.sort_index(ascending=True)

print(df_quarter)






# ############################################################################################################
# # YEARLY TO DATETIME - the ESG indicators 

df_year['Human Rights Controversies'] = df_year['Human Rights Controversies'].fillna(0)

# Define a function to convert 'FY-X' format to datetime
def convert_fy_to_datetime(fy_string):
    year = int(fy_string.split('-')[1])  # Extract the year from the 'FY-X' string
    calendar_year = 2023 - year  # Convert 'FY-X' year to calendar year
    return pd.Timestamp(year=calendar_year, month=3, day=31)  # Assuming FY starts in July

# Convert 'timestamp' column to datetime format
df_year['timestamp'] = df_year['timestamp'].apply(convert_fy_to_datetime)

# Set the datetime column as index
df_year.set_index('timestamp', inplace=True)

# print(df_year)

df_quarterly_interpolated = df_year.resample('Q').interpolate(method='linear') ## try different methods here (but not really keeping real form?), quadratric seems nice (EXCEPCT FOR CONTROVERSIES!!! )
df_quarterly_interpolated = df_quarterly_interpolated.sort_index(ascending=True)

print(df_quarterly_interpolated)

df_year = df_quarterly_interpolated




############################################################################################################################################
## HANDLE MISSING VALUE FOR BOTH DATAFRAMES SEPERATLY


df_ESG = df_year
df_financial = df_quarter



# Check for missing values after filling
for column in df_ESG.columns:
    missing_values = df_ESG[column].isnull().sum()
    print("Missing Values ESG:", column)
    print(missing_values)


# Check for missing values after filling
for column in df_financial.columns:
    missing_values = df_financial[column].isnull().sum()
    print("Missing Values financial:", column)
    print(missing_values)


## first interpolation to handle missing values within the datset

df_ESG = df_ESG.interpolate(method='linear')
df_financial = df_financial.interpolate(method = 'linear')

## second backward and forward fill to handle the missing values at beginning and end

# Fill missing values with forward fill
df_ESG = df_ESG.fillna(method='ffill')
df_filled = df_ESG.fillna(method='bfill')
df_ESG = df_filled

# Check for missing values after filling
missing_values_after_fill = df_ESG.isnull().sum()
print("Missing Values After Forward Fill:")
print(missing_values_after_fill)


# Fill missing values with forward fill
df_financial = df_financial.fillna(method='ffill')
df_filled = df_financial.fillna(method='bfill')
df_financial = df_filled

# Check for missing values after filling
missing_values_after_fill = df_financial.isnull().sum()
print("Missing Values After Forward Fill:")
print(missing_values_after_fill)


############################################
## check for missing values after missing value handling

# Check for missing values after filling
for column in df_ESG.columns:
    missing_values = df_ESG[column].isnull().sum()
    print("Missing Values ESG:", column)
    print(missing_values)


# Check for missing values after filling
for column in df_financial.columns:
    missing_values = df_financial[column].isnull().sum()
    print("Missing Values financial:", column)
    print(missing_values)


## SHOULD BE CONTAINING 0 MISSNG VALUES NOW

print(df_ESG)
print(df_financial)






#######################################################################################################################
# if there are booleans, make them into float

for column in df_ESG.columns:
    # Check if the column contains boolean values
    if df_ESG[column].dtype == 'bool':
        # Convert boolean values to float
        df_ESG[column] = df_ESG[column].astype(float)





# #####################################################################################################################
# # FEATURE ENGINEERING 


for column in df_ESG.columns:
    # Skip non-numeric columns if needed
    if not pd.api.types.is_numeric_dtype(df_ESG[column]):
        continue
    
    
    # Calculate the mean of every two consecutive values
    df_ESG[f'{column}_Mean'] = df_ESG[column].rolling(window=2).mean()
    
    # Calculate the rolling mean with window size 3
    df_ESG[f'{column}_Mean_3'] = df_ESG[column].rolling(window=3).mean()
    
    # Calculate the rolling mean with window size 10
    df_ESG[f'{column}_Mean_10'] = df_ESG[column].rolling(window=10).mean()
    
    # Calculate the percentage change between two consecutive values
    df_ESG[f'{column}_Percentage'] = df_ESG[column].pct_change()

    # DIFFERENCE
    diff_column_name = column + '_diff'
    df_ESG[diff_column_name] = df_ESG[column].diff()#.fillna(0)
    
    # INCREASE/DECREASE
    diff_cat_column_name = column + '_diff_cat'
    df_ESG[diff_cat_column_name] = df_ESG[diff_column_name].apply(lambda x: 1 if x > 0 else ((-1) if x < 0 else 0))

    # LAGS
    num_lags = 3  # Number of lagged features
    for i in range(1, num_lags + 1):
        df_ESG[f'{column}_lag_{i}'] = df_ESG[column].shift(i)

    # BINS (DIFFERENT RANGES?)
    if 'Score' in column:
        # Apply binning with bins of size 10 from 0 to 100
        bins = list(range(0, 101, 10))
        bin_column_name = column + '_bin'
        df_ESG[bin_column_name] = pd.cut(df_ESG[column], bins=bins, labels=False)
    elif column == 'CO2 Equivalent Emissions Total':
        # Apply binning with bins of size 100,000 from 0 to 10,000,000
        bins = list(range(0, 10000001, 100000))
        bin_column_name = column + '_bin'
        df_ESG[bin_column_name] = pd.cut(df_ESG[column], bins=bins, labels=False)



df_ESG['Environmental_Social_average'] = (df_ESG['Social Pillar Score'] + df_ESG['Environmental Pillar Score']) / 2
df_ESG['Environmental_Governance_average'] = (df_ESG['Governance Pillar Score'] + df_ESG['Environmental Pillar Score']) / 2
df_ESG['Social_Governance_average'] = (df_ESG['Governance Pillar Score'] + df_ESG['Social Pillar Score']) / 2

df_ESG['Human_Rights_Policy_interaction'] = df_ESG['Human Rights Policy'] * df_ESG['Social Pillar Score']
df_ESG['Human_Rights_Controversies_interaction'] = df_ESG['Human Rights Controversies'] * df_ESG['Social Pillar Score']

df_ESG['Eco_Design_Policy_interaction'] = df_ESG['Eco-Design Products'] * df_ESG['Environmental Pillar Score']
df_ESG['Bio_Diversity_Policy_interaction'] = df_ESG['Biodiversity Impact Reduction'] * df_ESG['Environmental Pillar Score']

df_ESG['CO2_Environmental_Interaction'] = df_ESG['CO2 Equivalent Emissions Total'] * df_ESG['Environmental Pillar Score']



########################################################################################
## SCALING

for column in df_ESG.columns:

    column_to_scale = df_ESG[column]
    # Compute the minimum and maximum values of the column
    min_value = column_to_scale.min()
    max_value = column_to_scale.max()
    df_ESG[column] = ((column_to_scale - min_value) / (max_value - min_value)) * (100 - 0) + 0 ## replace the column instead of adding a scaled column







# ###########################################
# ## handle missing value that come to light after creating new features

df_ESG = df_ESG.interpolate(method='linear')

# Fill missing values with forward fill
df_ESG = df_ESG.fillna(method='ffill')
df_filled = df_ESG.fillna(method='bfill')
df_ESG = df_filled

# Check for missing values after filling
missing_values_after_fill = df_ESG.isnull().sum()
print("Missing Values After Forward Fill:")
print(missing_values_after_fill)

# Check for missing values after filling
for column in df_ESG.columns:
    missing_values = df_ESG[column].isnull().sum()
    print("Missing Values ESG:", column)
    print(missing_values)

# Check for missing values after filling
for column in df_financial.columns:
    missing_values = df_financial[column].isnull().sum()
    print("Missing Values financial:", column)
    print(missing_values)




# ###############################################################################################################################
# WHEN STILL ANY MISSING VALUES -> REMOVE COLUMN

def remove_columns_with_missing_values(df):
    # Drop columns with any missing values
    df_cleaned = df.dropna(axis=1)
    return df_cleaned

# Remove columns with missing values
cleaned_df_ESG = remove_columns_with_missing_values(df_ESG)
cleaned_df_financial = remove_columns_with_missing_values(df_financial)

# Check for missing values after filling
for column in cleaned_df_ESG.columns:
    missing_values = cleaned_df_ESG[column].isnull().sum()
    print("Missing Values ESG:", column)
    print(missing_values)

# Check for missing values after filling
for column in cleaned_df_financial.columns:
    missing_values = cleaned_df_financial[column].isnull().sum()
    print("Missing Values financial:", column)
    print(missing_values)


df_ESG = cleaned_df_ESG
df_financial = cleaned_df_financial








# #########################################################################################################
# ## MERGE THE DATAFRAMES 


# Extract year and month from the timestamp for both DataFrames
df_financial['year_month'] = df_financial.index.to_period('M')
df_ESG['year_month'] = df_ESG.index.to_period('M')

# # Merge DataFrames based on the year_month column
merged_df = pd.merge(left=df_financial, right=df_ESG, on='year_month', how='left')
print(merged_df)

# Drop the year_month column if not needed
merged_df.drop('year_month', axis=1, inplace=True)

print(merged_df)

# print(merged_df.describe())

#####################################
## HANDLE MISSING VALUES AFTER MERGING 

# Fill missing values with forward fill
merged_df = merged_df.fillna(method='ffill')
df_filled = merged_df.fillna(method='bfill')
merged_df = df_filled

## Check for missing values after filling
for column in merged_df.columns:
    missing_values = merged_df[column].isnull().sum()
    print("Missing Values merged df:", column)
    print(missing_values)




#####################################################################################################################################################################
## LINEAR REGRESSION

##############################################################################################################################################################
## backward selection method LR in loop

df = merged_df

numeric_df = df.select_dtypes(include=[np.number]) ## make categorical features into numerical features to be able to use them here 

# Assuming X contains your feature matrix and y contains your target variable
X = numeric_df.drop(columns=['Earnings Per Share Reported - Actual', 'Return On Equity - Actual', 'Return On Assets - Actual', 'Earnings Per Share - Actual'])  # Features
y = numeric_df[target_variable]  # Target variable

# Initialize arrays to store the errors
rmse_errors = []

# Variables to store the best number of features, the lowest RMSE, best features, and best hyperparameters
best_nr_features = None
lowest_rmse = float('inf')
best_features = []

# Loop over the number of features to select
for n_features in range(1, num_features):
    # Initialize the Linear Regression model
    model = LinearRegression()

    # Initialize Recursive Feature Elimination (RFE) with the Linear Regression model
    rfe = RFE(estimator=model, n_features_to_select=n_features)

    # Fit RFE to the data
    rfe.fit(X, y)

    # Get the selected features
    selected_features = X.columns[rfe.support_]

    # Fit the Linear Regression model on the selected features
    model.fit(X[selected_features], y)

    # Predict the target variable using the model
    y_pred = model.predict(X[selected_features])

    # Calculate the RMSE and MAE for the model
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Store the errors
    rmse_errors.append(rmse)

    if rmse < lowest_rmse:
        lowest_rmse = rmse
        best_nr_features = n_features
        best_features = selected_features


# Plot the errors against the number of features
plt.figure(figsize=(12, 6))
plt.plot(range(1, num_features), rmse_errors, marker='o')
plt.xlabel('Number of Selected Features')
plt.ylabel('Root Mean Squared Error')
plt.title('EPSR - Error vs. Number of Selected Features Linear Regression') ## adjust
plt.grid(True)
# plt.show()
plt.tight_layout() 

# Save plot to file
plt.savefig(f'EPSR_errors_linear_regression.png')  # adjust 
plt.close()

lowest_rmse_LR = lowest_rmse
best_nr_features_LR = best_nr_features
best_features_LR = best_features

print('LR best nr features', best_nr_features_LR)

## list of best ones + their importance
model = LinearRegression()

# Initialize Recursive Feature Elimination (RFE) with the Linear Regression model
rfe = RFE(estimator=model, n_features_to_select=best_nr_features_LR)

# Fit RFE to the data
rfe.fit(X, y)

# Get the selected features
selected_features = X.columns[rfe.support_]

# Fit the Linear Regression model on the selected features
model.fit(X[selected_features], y)

# Get the coefficients from the fitted model
coefficients = model.coef_

# Create a DataFrame for the feature importances (absolute value of coefficients)
feature_importances = pd.DataFrame({
    'Feature': selected_features,
    'Importance': np.abs(coefficients)
})

# Sort the features by importance
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Plot the feature importance of best nr of features
plt.figure(figsize=(12, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title(f'EPSR - Feature Importances Linear Regression (n_features={best_nr_features_LR})') ## adjust
plt.tight_layout() 

# Save plot to file
plt.savefig(f'EPSR_feature_importance_linear_regression.png')  # adjust
plt.close()

best_features_LR = feature_importances['Feature']
print('best features LR sorted:')
print(best_features_LR)









#############################################################################################################################################################
## RANDOM FOREST

###############################################################################################################################################################
## backward selection method Random Forest (RF)

numeric_df = df

X = numeric_df.drop(columns=['Earnings Per Share Reported - Actual', 'Return On Assets - Actual', 'Return On Equity - Actual', 'Earnings Per Share - Actual'])
y = numeric_df[target_variable]

# Initialize an array to store the errors
errors_RMSE = []

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

best_nr_features = None
lowest_rmse = float('inf')
best_features = []
best_hyperparameters = {}

# Loop over the number of features to select
for n_features in range(1, num_features):
    # Initialize the Random Forest regressor model
    model = RandomForestRegressor()

    # Initialize Recursive Feature Elimination (RFE) with the Random Forest model
    rfe = RFE(estimator=model, n_features_to_select=n_features)

    # Fit RFE to the data
    rfe.fit(X, y)

    # Get the selected features
    selected_features = X.columns[rfe.support_]

    # Perform grid search with cross-validation to find the best parameters
    grid_search = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X[selected_features], y)

    # Get the best model from the grid search
    best_model = grid_search.best_estimator_

    # Fit the best model on the selected features
    best_model.fit(X[selected_features], y)

    # Calculate the mean squared error for the model
    y_pred = best_model.predict(X[selected_features])

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    # mae = mean_absolute_error(y, y_pred)

    errors_RMSE.append(rmse)

    # Update the best number of features if the current RMSE is lower than the lowest RMSE
    if rmse < lowest_rmse:
        lowest_rmse = rmse
        best_nr_features = n_features
        best_features = selected_features
        best_hyperparameters = grid_search.best_params_


# Plot the errors against the number of features
plt.figure(figsize=(12, 6))
plt.plot(range(1, num_features), errors_RMSE, marker='o')
plt.xlabel('Number of Selected Features')
plt.ylabel('Root Mean Squared Error')
plt.title('EPSR - Error vs. Number of Selected Features Random Forest') ## adjust
plt.grid(True)
# plt.show()
plt.tight_layout() 

# Save plot to file
plt.savefig(f'EPSR_errors_random_forest.png')  # adjust
plt.close()

lowest_rmse_RF = lowest_rmse
best_nr_features_RF = best_nr_features
best_features_RF = best_features

model = RandomForestRegressor()

# Initialize Recursive Feature Elimination (RFE) with the Random Forest model
rfe = RFE(estimator=model, n_features_to_select=best_nr_features_RF)

# Fit RFE to the data
rfe.fit(X, y)

# Get the selected features
selected_features = X.columns[rfe.support_]

# Perform grid search with cross-validation to find the best parameters
grid_search = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X[selected_features], y)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Fit the best model on the selected features
best_model.fit(X[selected_features], y)

# Get the feature importances from the fitted model
importances = best_model.feature_importances_

# Create a DataFrame for the feature importances
feature_importances = pd.DataFrame({
    'Feature': selected_features,
    'Importance': importances
})

# Sort the features by importance
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(12, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title(f'EPSR - Feature Importances Random Forest (n_features={best_nr_features_RF})') ## adjust
# plt.show()
plt.tight_layout() 

# Save plot to file
plt.savefig(f'EPSR_feature_importance_random_forest.png')  # adjust
plt.close()

best_features_RF = feature_importances['Feature']
print('best features RF sorted:')
print(best_features_RF)





#################################################################################################
## EVALUATION AND COMPARISION OF MODELS 


# Evaluation Metrics for Linear Regression
print(f'({target_variable}) Evaluation Metrics for Linear Regression:')
print(f'{"Metric":<45}  {"Best after Feature Selection":<30}')
print(f'{"Root Mean Squared Error":<45}  {lowest_rmse_LR:<30}')

# Evaluation Metrics for Random Forest
print(f'\n({target_variable}) Evaluation Metrics for Random Forest:')
print(f'{"Metric":<45}  {"Best after Feature Selection":<30}')
print(f'{"Root Mean Squared Error":<45} {lowest_rmse_RF:<30}')

print('Optimal nr of features + list of most important feature Linear Regression:')
print(best_nr_features_LR)
print(best_features_LR)

print('Optimal nr of features + list of most important feature Random Forest:')
print(best_nr_features_RF)
print(best_features_RF)


# # intersection of the two feature lists
# intersection_features = best_features_LR.intersection(best_features_RF)

# print('intersection of best features from Linear Regression and Random Forest:')
# print(intersection_features)

# number_intersection = len(intersection_features)
# print('number of features in the intersection:', number_intersection)


# Convert Series to sets
best_features_LR_set = set(best_features_LR)
best_features_RF_set = set(best_features_RF)

# Find the intersection of the two sets
intersection_features = best_features_LR_set.intersection(best_features_RF_set)

print("Intersection of features:", intersection_features)

# number_intersection = len(intersection_features)
# print('number of features in the intersection:', number_intersection)


# print('tuned HP selectiom', best_hyperparameters)
# print('best nr of features for selection', best_nr_features)
# print('tuned HP all', best_params)





