import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor


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




#####################################################################################################################
## DATA INVESTIGATION

#####################################################################################################################
## NUMERICAL INVESTIGATION 

print('describe ESG')
print(df_year['CO2 Equivalent Emissions Total'].describe())
print('describe financial')
print(df_quarter.describe())





# Assuming 'df' is your DataFrame
missing_values_count = df_year.isna().sum()
print("Missing values count per column for ESG data:")
print(missing_values_count)

# Assuming 'df' is your DataFrame
missing_values_count = df_quarter.isna().sum()
print("Missing values count per column for financial data:")
print(missing_values_count)



# ######################################################################################################################
# # VISUAL INVESTIGATION


# SCATTERPLOT


for i in range(1, len(df_year.columns)):
    plt.figure(figsize=(10, 6))
    plt.scatter(df_year.index, df_year.iloc[:, i], color='blue')  # 's' controls the marker size
    plt.title(f'Scatter Plot of {df_year.columns[i]}')
    plt.xlabel('Timestamp')
    plt.ylabel(df_year.columns[i])

plt.show()


for i in range(1, len(df_quarter.columns)):
    plt.figure(figsize=(10, 6))
    plt.scatter(df_quarter.index, df_quarter.iloc[:, i], s=10, color='blue')  # 's' controls the marker size
    plt.title(f'Scatter Plot of {df_quarter.columns[i]}')
    plt.xlabel('Timestamp')
    plt.ylabel(df_quarter.columns[i])

plt.show()



##############################################################################################
# HISTOGRAM

for i in range(1, len(df_year.columns)):
    plt.figure(figsize=(10, 6))
    sns.histplot(df_year.iloc[:, i])
plt.show()

for i in range(1, len(df_quarter.columns)):
    plt.figure(figsize=(10, 6))
    sns.histplot(df_quarter.iloc[:, i])
plt.show()



########################################################################################
# BOXPLOT


for i, column in enumerate(df_year.columns[1:], start=1):
    plt.figure(figsize=(10, 6))
    df_year.boxplot(column=[column])
    plt.title(f'Boxplot of {column}')
    plt.ylabel(column)
    plt.xticks(rotation=45)
    plt.tight_layout()

plt.show()


for i, column in enumerate(df_quarter.columns[1:], start=1):
    plt.figure(figsize=(10, 6))
    df_quarter.boxplot(column=[column])
    plt.title(f'Boxplot of {column}')
    plt.ylabel(column)
    plt.xticks(rotation=45)
    plt.tight_layout()

plt.show()







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











# # ##################################################################################################################
# # OUTLIER DETECTION

# Iterate over each feature in df_ESG
for column in df_ESG.columns[1:]:
    # Reshape the data for LOF input (needs 2D array)
    X = df_ESG[column].values.reshape(-1, 1)

    # Fit the LOF model to the data
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
    y_pred = lof.fit_predict(X)

    # Plot the data with outliers highlighted
    plt.figure(figsize=(10, 6))
    plt.scatter(df_ESG.index, df_ESG[column], c='blue', label='Data')
    plt.scatter(df_ESG.index[y_pred == -1], df_ESG[column][y_pred == -1], c='red', label='Outliers')
    plt.title(f'Identifying Outliers for "{column}" using Local Outlier Factor (LOF)')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    plt.show()



# Iterate over each feature in df_ESG
for column in df_financial.columns[1:]:
    # Reshape the data for LOF input (needs 2D array)
    X = df_financial[column].values.reshape(-1, 1)

    # Fit the LOF model to the data
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
    y_pred = lof.fit_predict(X)

    # Plot the data with outliers highlighted
    plt.figure(figsize=(10, 6))
    plt.scatter(df_financial.index, df_financial[column], c='blue', label='Data')
    plt.scatter(df_financial.index[y_pred == -1], df_financial[column][y_pred == -1], c='red', label='Outliers')
    plt.title(f'Identifying Outliers for "{column}" using Local Outlier Factor (LOF)')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    plt.show()



# #############################
# ## did not take a look into simple distance based since there you compare with all datapoints
# ## since it is more about the progression of the points (time series), only surrounding points are interesting (see Mark his book)





###############################################################################################################
## VISUALIZATION NOW THAT IT IS TIME SERIES DATA


for i in range(0, len(df_year.columns)):
    plt.figure(figsize=(10, 6))
    plt.scatter(df_year.index, df_year.iloc[:, i], c='blue', label='Data')
    
    plt.title(f'Scatter Plot of {df_year.columns[i]}')
    plt.xlabel('Timestamp')
    plt.ylabel(df_year.columns[i])
    
plt.show()

for i in range(0, len(df_quarter.columns)):
    plt.figure(figsize=(10, 6))
    plt.scatter(df_quarter.index, df_quarter.iloc[:, i], c='blue', label='Data')
    
    plt.title(f'Scatter Plot of {df_quarter.columns[i]}')
    plt.xlabel('Timestamp')
    plt.ylabel(df_quarter.columns[i])
    
plt.show()









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
# print(df_ESG)
# print(df_financial)







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

print(df)



