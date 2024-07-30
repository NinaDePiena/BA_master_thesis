import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ARIMA

from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import warnings


# Suppress specific deprecation warning
warnings.filterwarnings("ignore", category=FutureWarning, message="'squared' is deprecated in version 1.4 and will be removed in 1.6. To calculate the root mean squared error, use the function 'root_mean_squared_error'.")

# VARIABLES 

alpha = 0.05
number_of_folds = 5
p_down = 0 # AR, PACF
d_down = 0
q_down = 0 # MA, ACF
p_up = 15 # AR, PACF
d_up = 5
q_up = 10 # MA, ACF


target_var = 'Earnings Per Share - Actual' # adjust
exog_var = 'ESG Combined Score' # adjust



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


#######################################################################################################################
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


print(df_ESG)



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


df = merged_df









# ########################################################################################################################
## K FOLD CROSS VALIDATION ARIMA

data = merged_df


## model evaluation 

def arima_model_evaluate(train_data, test_data, order):
    try:
        # Model initialize and fit
        ARIMA_model = ARIMA(train_data, order=order)
        ARIMA_model = ARIMA_model.fit()
        # Getting the predictions
        predictions = ARIMA_model.get_forecast(len(test_data.index))
        predictions_df = predictions.conf_int(alpha=0.05)
        predictions_df["Predictions"] = ARIMA_model.predict(start=predictions_df.index[0], end=predictions_df.index[-1])
        predictions_df.index = test_data.index
        predictions_arima = predictions_df["Predictions"]
        # calculate MSE score
        mse_score = mean_squared_error(test_data[0:].values, predictions_df["Predictions"]) ## test_data[0:] ????
        return mse_score
    except np.linalg.LinAlgError as e:
        print(f"LinAlgError for ARIMA order {order}: {e}")
        return float("inf")
    except ValueError as e:
        print(f"ValueError for ARIMA order {order}: {e}")
        return float("inf")


def arimax_model_evaluate(train_data, test_data, order, exog_train, exog_test):
    try:
        # Model initialize and fit
        ARIMAX_model = ARIMA(train_data, order=order, exog=exog_train, trend='n') 
        ARIMAX_model = ARIMAX_model.fit()
        # Getting the predictions
        predictions = ARIMAX_model.get_forecast(steps=len(test_data), exog=exog_test)
        predictions_df = predictions.conf_int(alpha=0.05)
        predictions_df["Predictions"] = ARIMAX_model.forecast(steps=len(test_data), exog=exog_test)
        predictions_df.index = test_data.index
        predictions_arima = predictions_df["Predictions"]
        # calculate RMSE score
        mse_score = mean_squared_error(test_data.values, predictions_df["Predictions"])
        return mse_score
    except np.linalg.LinAlgError as e:
        print(f"LinAlgError for ARIMAX order {order}: {e}")
        return float("inf")
    except ValueError as e:
        print(f"ValueError for ARIMAX order {order}: {e}")
        return float("inf")


def evaluate_models_ARIMA(train_data, test_data, list_p_values, list_d_values, list_q_values):
    best_mse, best_config = float("inf"), None
    for p in list_p_values:
        for d in list_d_values:
            for q in list_q_values:
                arima_order = (p,d,q)
                mse = arima_model_evaluate(train_data, test_data, arima_order)
                if mse < best_mse:
                    best_mse, best_config = mse, arima_order
                    print('ARIMA%s MSE=%.5f' % (arima_order,mse))
                    print('Best Configuration: ARIMA%s , MSE=%.5f' % (best_config, best_mse))
    return best_config

def evaluate_models_ARIMAX(train_actuals, test_data, list_p_values, list_d_values, list_q_values, exog_train, exog_test):
    best_mse, best_config = float("inf"), None
    for p in list_p_values:
        for d in list_d_values:
            for q in list_q_values:
                arima_order = (p, d, q)
                mse = arimax_model_evaluate(train_actuals, test_data, arima_order, exog_train, exog_test)
                if mse < best_mse:
                    best_mse, best_config = mse, arima_order
                    print('ARIMA%s MSE=%.3f' % (arima_order, mse))
                    print('Best Configuration x: ARIMA%s , MSE=%.3f' % (best_config, best_mse))
    return best_config



target_variable = target_var
exogenous_variables = [exog_var]


data = df[target_variable]
exogenous_data = df[exogenous_variables]

# Prepare TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=number_of_folds)

print('dataframe for target variable')
print(data)

print('dataframe for exogenous variable')
print(exogenous_data)

rmse_arima = []
rmse_arimax = []
best_config_ARIMA = None
best_config_ARIMAX = None

fold = 1

for train_index, test_index in tscv.split(data):
    print('train index', train_index)
    print('test index', test_index)
    train_data, test_data = data.iloc[train_index], data.iloc[test_index]
    print('train data', train_data)
    print('test data', test_data)
    exog_train, exog_test = exogenous_data.iloc[train_index], exogenous_data.iloc[test_index]

    # Find best parameters for ARIMA model
    p_values = range(p_down, p_up)
    d_values = range(d_down, d_up)
    q_values = range(q_down, q_up)
    best_config_ARIMAX = evaluate_models_ARIMAX(train_data, test_data, p_values, d_values, q_values, exog_train, exog_test)
    best_config_ARIMA = evaluate_models_ARIMA(train_data, test_data, p_values, d_values, q_values)

    # Fit ARIMA model using the best configuration
    try:
        ARIMA_model = ARIMA(train_data, order=best_config_ARIMA)
        ARIMA_model = ARIMA_model.fit()
        predictions_arima = ARIMA_model.predict(start=test_data.index[0], end=test_data.index[-1])
        rmse_arima.append(mean_squared_error(test_data, predictions_arima, squared=False))
    except Exception as e:
        print(f"Failed to fit ARIMA model with order {best_config_ARIMA}: {e}")
        rmse_arima.append(float("inf"))

    # Fit ARIMAX model using the best configuration
    try:
        ARIMAX_model = ARIMA(train_data, order=best_config_ARIMAX, exog=exog_train)
        ARIMAX_model = ARIMAX_model.fit()
        predictions_arimax = ARIMAX_model.predict(start=test_data.index[0], end=test_data.index[-1], exog=exog_test)
        rmse_arimax.append(mean_squared_error(test_data, predictions_arimax, squared=False))
    except Exception as e:
        print(f"Failed to fit ARIMAX model with order {best_config_ARIMAX}: {e}")
        rmse_arimax.append(float("inf"))

    # Plotting fold 
    plt.figure(figsize=(10, 6))
    plt.plot(train_data, color="black", label='Train')
    plt.plot(test_data, color="green", label='Test')
    plt.xticks(rotation=35)
    plt.title(f'EPS - ARIMA VS ARIMAX - Fold {fold}') # adjust
    plt.plot(test_data.index, predictions_arima, color="red", label='ARIMA Predictions')
    plt.plot(test_data.index, predictions_arimax, color="blue", label='ARIMAX Predictions')
    plt.legend()
    # plt.show()

    # Save plot to file
    plt.savefig(f'MS_EPS_ARIMA_combined_fold{fold}.png')  # adjust
    plt.close()
    
    fold += 1







###########################################################################################################
## EVALUATION


# Plotting one example (last fold)
plt.figure(figsize=(10, 6))
plt.plot(train_data, color="black", label='Train')
plt.plot(test_data, color="green", label='Test')
plt.xticks(rotation=35)
plt.title("EPS - ARIMA VS ARIMAX - Last Fold") ## adjust
plt.xlabel('Time')

# Predictions from the last fold
plt.plot(test_data.index, predictions_arima, color="red", label='ARIMA Predictions')
plt.plot(test_data.index, predictions_arimax, color="blue", label='ARIMAX Predictions')
plt.legend()
# plt.show()

 # Save plot to file
plt.savefig('MS_EPS_ARIMA_combined_finalfold.png') ## adjust
plt.close()
    

print(rmse_arima)
print(rmse_arimax)

# Print average RMSE values
print(f"Average RMSE for ARIMA: {np.mean(rmse_arima)}")
print(f"Average RMSE for ARIMAX: {np.mean(rmse_arimax)}")

# Print the final best configurations for ARIMA and ARIMAX
print(f"Best configuration for ARIMA: {best_config_ARIMA}")
print(f"Best configuration for ARIMAX: {best_config_ARIMAX}")