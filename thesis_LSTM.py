import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error
import warnings


# LSTM

from sklearn.model_selection import ParameterGrid
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from itertools import product
from sklearn.model_selection import TimeSeriesSplit



# Suppress specific deprecation warning
warnings.filterwarnings("ignore", category=FutureWarning, message="'squared' is deprecated in version 1.4 and will be removed in 1.6. To calculate the root mean squared error, use the function 'root_mean_squared_error'.")

number_of_folds = 5

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


#################################################################################################
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





## LSTM


#############################################################################################
## BASE MODEL LSTM

data = df

# Assuming data contains your time series data
target_variable = data[target_var].values

# Normalize the target variable
scaler = MinMaxScaler(feature_range=(0, 1))
target_variable = scaler.fit_transform(target_variable.reshape(-1, 1))

# Prepare the data for LSTM
def prepare_data_for_lstm(data, n_steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data) - 1:
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix] 
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Choose the number of time steps ## is this sequence length?
n_steps = 3

# Prepare data for LSTM
X, y = prepare_data_for_lstm(target_variable, n_steps)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Split data into train and test sets
train_size = int(len(X) * 0.8)  # 80% train, 20% test
test_size = len(X) - train_size
train_X, test_X = X[:train_size], X[train_size:]
train_y, test_y = y[:train_size], y[train_size:]

# Define LSTM model # github code
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = self.fc(out[:, -1, :])
        return out

# Define hyperparameters
input_size = 1
hidden_size = 50
num_layers = 1
learning_rate = 0.001
num_epochs = 100
batch_size = 32

# Instantiate the model
model = LSTMModel(input_size, hidden_size, num_layers)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model # github code
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(train_X)
    loss = criterion(outputs, train_y)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Train - Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


## how is this different than just the loss in the above epoch loop?
model.eval() ## what does this function do?
test_outputs = model(test_X)
test_loss = criterion(test_outputs, test_y)
print(f'Test - Loss: {test_loss.item():.4f}')

predicted_test = model(test_X).detach().numpy()

# Inverse transform the test predictions and actual values
predicted_test = scaler.inverse_transform(predicted_test)
actual_test = scaler.inverse_transform(test_y.numpy())

# Plot the results including train and test data
plt.plot(data.index[n_steps:train_size + n_steps], scaler.inverse_transform(y[:train_size].numpy()), color='black', label='Train')
plt.plot(data.index[train_size + n_steps:], scaler.inverse_transform(test_y.numpy()), color='green', label='Test')
plt.plot(data.index[train_size + n_steps:], predicted_test, color='red', label='Predicted')
plt.xlabel('Time')
plt.ylabel('Target Variable')
plt.title('LSTM base model')
plt.legend()
# plt.show()

# Calculate Mean Squared Error (RMSE) for test data
rmse_base_LSTM = mean_squared_error(actual_test, predicted_test, squared=False)







###########################################################################################
## LSTM WITH CROSS FOLD AND EXOGENOUS VARIABLE


## model without exog

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = self.fc(out[:, -1, :])
        return out
    


## model with exog

class LSTMExogModel(nn.Module):
    def __init__(self, input_size, exog_size, hidden_size, num_layers=1):
        super(LSTMExogModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size + exog_size, 1)  # Add the size of the exogenous variable to the LSTM output size
        
    def forward(self, x, exog):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = torch.cat((out[:, -1, :], exog), dim=1)  # Concatenate LSTM output with exogenous variable
        out = self.fc(out)
        return out

            

# Prepare the data for LSTM with exogenous variable ## previous code, one below is from chat 
def prepare_data_for_lstmx(data, exog_data, n_steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data) - 1:
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y), exog_data[n_steps:]



###############################################################################################
## LSTM MODEL WITH CROSSFOLD 


# Prepare TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=number_of_folds)
n_steps = 3
rmse_lstm = []
rmse_lstmx = []

def def_best_param_lstm(X, y):
    # Define the hyperparameters to tune
    # param_grid = {
    #     'hidden_size': [100, 150],
    #     'num_layers': [1, 2],
    #     'learning_rate': [0.001, 0.01],
    #     'num_epochs': [100, 150],
    #     'batch_size': [32, 64]
    # }

    param_grid = {
    'hidden_size': [75, 100, 150, 175],
    'num_layers': [1, 2, 3],
    'learning_rate': [0.001, 0.005, 0.01],
    'num_epochs': [100, 125, 150],
    'batch_size': [32, 48, 64]
    }

    # Generate all possible combinations of hyperparameters
    param_combinations = ParameterGrid(param_grid)

    best_rmse = float('inf')
    best_params = None

    # Iterate over all parameter combinations
    for params in param_combinations:
        # Instantiate the model
        model = LSTMModel(input_size, params['hidden_size'], params['num_layers'])
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        # Train the model
        for epoch in range(params['num_epochs']):
            # Shuffle and batch the data
            indices = torch.randperm(X.size(0))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            for i in range(0, X.size(0), params['batch_size']):
                batch_X = X_shuffled[i:i+params['batch_size']]
                batch_y = y_shuffled[i:i+params['batch_size']]
                
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Evaluate the model
        with torch.no_grad():
            predicted = model(X).numpy()
        
        predicted = scaler.inverse_transform(predicted)
        actual = scaler.inverse_transform(y.numpy())
        
        rmse = mean_squared_error(actual, predicted, squared=False) ## is this indeed the right formula????v
        
        # Update best parameters if MSE is lower
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params

    print("Best parameters:", best_params)
    print("Best RMSE:", best_rmse)

    return best_params




def def_best_param_lstmx(train_X, test_X, train_y, test_y, train_exog, test_exog, input_size, exog_size):
    # Define the hyperparameters to tune
    # param_grid = {
    #     'hidden_size': [100, 150],
    #     'num_layers': [1, 2],
    #     'learning_rate': [0.001, 0.01],
    #     'num_epochs': [100, 150],
    #     'batch_size': [32, 64]
    # }

    param_grid = {
    'hidden_size': [75, 100, 150, 175],
    'num_layers': [1, 2, 3],
    'learning_rate': [0.001, 0.005, 0.01],
    'num_epochs': [100, 125, 150],
    'batch_size': [32, 48, 64]
    }

        
    # Generate all possible combinations of hyperparameters
    param_combinations = product(param_grid['hidden_size'], param_grid['num_layers'],
                                param_grid['learning_rate'], param_grid['num_epochs'], param_grid['batch_size'])

    best_rmse = float('inf')
    best_paramsx = None

    # Iterate over all parameter combinations
    for hidden_size, num_layers, learning_rate, num_epochs, batch_size in param_combinations:
        # Instantiate the model
        
        model = LSTMExogModel(input_size, exog_size, hidden_size, num_layers)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Train the model
        for epoch in range(num_epochs):
            # Forward pass
            outputs = model(train_X, train_exog)
            loss = criterion(outputs, train_y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Make predictions on test data
        predicted_test = model(test_X, test_exog).detach().numpy()
        
        # Inverse transform the test predictions and actual values
        predicted_test = scaler.inverse_transform(predicted_test)
        actual_test = scaler.inverse_transform(test_y.numpy())
        
        # Calculate MSE for test data
        rmse = mean_squared_error(actual_test, predicted_test, squared=False)
        
        # Update best parameters if MSE is lower
        if rmse < best_rmse:
            best_rmse = rmse
            best_paramsx = {'hidden_size': hidden_size, 'num_layers': num_layers,
                        'learning_rate': learning_rate, 'num_epochs': num_epochs, 'batch_size': batch_size}

    print("Best parameters:", best_paramsx)
    print("Best RMSE:", best_rmse)

    return best_paramsx



## loop though different folds of the crossfold 

fold = 1

for train_index, test_index in tscv.split(data):

    target_variable = data[target_var].values
    exog_variable = data[exog_var].values ## SINGLE
    # exog_variables = data[exog_vars].values  # Example with multiple exogenous variables ## MULTIPLE


    # Normalize variables
    scaler = MinMaxScaler(feature_range=(0, 1))
    target_variable = scaler.fit_transform(target_variable.reshape(-1, 1))
    # Normalize the exogenous variable
    scaler_exog = MinMaxScaler(feature_range=(0, 1))
    exog_variable_scaled = scaler_exog.fit_transform(exog_variable.reshape(-1, 1))
    # exog_variables_scaled = scaler_exog.fit_transform(exog_variables) ## MULTIPLE (whty not the reshape??)

    # Prepare data for LSTM
    X, y = prepare_data_for_lstm(target_variable, n_steps)
    X, y, exog_data = prepare_data_for_lstmx(target_variable, exog_variable_scaled, n_steps) ## SINGLE
    # X, y, exog_data = prepare_data_for_lstmx(target_variable, exog_variables_scaled, n_steps) ## MULTIPLE


    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    exog_data = torch.tensor(exog_data, dtype=torch.float32)  # Add an extra dimension to make it 2D


    print('lenght data LSTM', len(data))
    train_size = len(train_index)
    print('train size LSTM', train_size)
    test_size = len(test_index)
    print('test size LSTM', test_size)


   # Create the train and test sets with the specified sizes
    train_X, test_X = X[:train_size], X[train_size:train_size + test_size]
    train_y, test_y = y[:train_size], y[train_size:train_size + test_size]
    train_exog, test_exog = exog_data[:train_size], exog_data[train_size:train_size + test_size]


    print('data train X lstm', train_X)
    print('data test X lstm', test_X)
    print('data train Y lstm', train_y)
    print('data test Y lstm', test_y)
    print('data train exog lstm', train_exog)
    print('data test exog lstm', test_exog)


    exog_size = 1 ## single var, make different in the case multiple exogenous variables 
    input_size = 1 ## SINGLE
    # exog_size = exog_variables.shape[1] ## MULTIPLE 

    best_params = def_best_param_lstm(X, y)
    best_paramsx = def_best_param_lstmx(train_X, test_X, train_y, test_y, train_exog, test_exog, input_size, exog_size)
   

    print('best params function', best_params)
    print('best params with exog function', best_paramsx)

    # LSTM without exog 

    ## Instantiate the model with the best parameters
    model = LSTMModel(input_size, best_params['hidden_size'], best_params['num_layers'])

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])

    # Train the model with the best parameters
    for epoch in range(best_params['num_epochs']):
        # Forward pass
        outputs = model(train_X)
        loss = criterion(outputs, train_y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Convert the model to evaluation mode
    model.eval()

    # Make predictions on test data
    with torch.no_grad():
        predicted_test_LSTM = model(test_X).numpy()

    # Inverse transform the test predictions and actual values
    predicted_test_LSTM = scaler.inverse_transform(predicted_test_LSTM)
    actual_test = scaler.inverse_transform(test_y.numpy())
    # actual_test = scaler_target.inverse_transform(test_y.numpy().reshape(-1, 1)) # from chat ## IS THIS FOR MUKTPLE??


    # Calculate Mean Squared Error (MSE) for test data

    rmse_tuned_LSTM = mean_squared_error(actual_test, predicted_test_LSTM, squared=False)
    print('RMSE for models', rmse_tuned_LSTM)
    rmse_lstm.append(rmse_tuned_LSTM)


    ## LSTM with exogenous variable

    model = LSTMExogModel(input_size, exog_size, best_params['hidden_size'], best_params['num_layers'])

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])

    # Train the model with the best parameters
    for epoch in range(best_params['num_epochs']):
        # Forward pass
        outputs = model(train_X, train_exog)
        loss = criterion(outputs, train_y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Convert the model to evaluation mode
    model.eval()

    # # Make predictions on test data
    # with torch.no_grad():
    #     predicted_test_LSTMX = model(test_X, test_exog).numpy()

    # Make predictions on test data
    predicted_test_LSTMX = model(test_X, test_exog).detach().numpy()

    # Inverse transform the test predictions and actual values
    predicted_test_LSTMX = scaler.inverse_transform(predicted_test_LSTMX)
    actual_test = scaler.inverse_transform(test_y.numpy())



    # Calculate Mean Squared Error (MSE) for test data
    rmse_tuned_LSTMX = mean_squared_error(actual_test, predicted_test_LSTMX, squared=False)
    print('RMSE for models', rmse_tuned_LSTMX)
    rmse_lstmx.append(rmse_tuned_LSTMX)

    ## plot lstm and lstmx together 

    # Plot the results including train and test
    plt.figure(figsize=(10, 6))
    plt.plot(data.index[n_steps:train_size + n_steps], scaler.inverse_transform(y[:train_size].reshape(-1, 1)), label='Train', color="black")
    plt.plot(data.index[train_size + n_steps: train_size + n_steps + test_size], actual_test, label='Test', color="green")
    plt.plot(data.index[train_size + n_steps: train_size + n_steps + test_size], predicted_test_LSTM, label='LSTM Predictions', color="red")
    plt.plot(data.index[train_size + n_steps: train_size + n_steps + test_size], predicted_test_LSTMX, label='LSTM exog Predictions', color="blue")
    plt.xlabel('Time')
    plt.title(f'EPS - LSTM VS LSTM with exogenous variable models - Fold {fold}')  # adjust
    plt.legend()
    # plt.show()

     # Save plot to file
    plt.savefig(f'MS_EPS_LSTM_base_fold{fold}.png')  # adjust
    plt.close()
    
    fold += 1





#########################################################################################
## EVALUATION


# Plot the results including train and test
plt.figure(figsize=(10, 6))
plt.plot(data.index[n_steps:train_size + n_steps], scaler.inverse_transform(y[:train_size].reshape(-1, 1)), label='Train', color="black")
plt.plot(data.index[train_size + n_steps: train_size + n_steps + test_size], actual_test, label='Test', color="green")
plt.plot(data.index[train_size + n_steps: train_size + n_steps + test_size], predicted_test_LSTM, label='LSTM Predictions', color="red")
plt.plot(data.index[train_size + n_steps: train_size + n_steps + test_size], predicted_test_LSTMX, label='LSTM exog Predictions', color="blue")
plt.xlabel('Time')
plt.title('EPS - LSTM VS LSTM with exogenous variable models - Last Fold') # adjust
plt.legend()
# plt.show()

# Save plot to file
plt.savefig(f'MS_EPS_LSTM_base_finalfold.png')  # adjust
plt.close()


## RESULTS 

print('MS - EPS - LSTM - base - big param') ## adjust

print(rmse_lstm)
print(rmse_lstmx)

# Print average RMSEs for reference
print(f"Average RMSE for LSTM: {np.mean(rmse_lstm)}")
print(f"Average RMSE for LSTMX: {np.mean(rmse_lstmx)}")

print('Best parameters LSTM', best_params)
print('Best parameters LSTMX', best_paramsx)
