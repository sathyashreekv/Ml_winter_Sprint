import pandas as pd
import numpy as np

"""
PANDAS FOR MACHINE LEARNING - ESSENTIAL OPERATIONS
From data loading to feature engineering - everything you need
"""

print("="*60)
print("1. DATA CREATION & LOADING - Foundation")
print("="*60)

# Creating DataFrames (multiple ways)
data_dict = {
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'age': [25, 30, 35, 28, 32],
    'salary': [50000, 60000, 70000, 55000, 65000],
    'department': ['Engineering', 'Marketing', 'Engineering', 'HR', 'Marketing'],
    'experience': [2, 5, 8, 3, 6]
}

df = pd.DataFrame(data_dict)
print("DataFrame from dictionary:")
print(df)
print(f"Shape: {df.shape}")
print(f"Data types:\n{df.dtypes}")

# Essential info about your data
print(f"\nDataFrame Info:")
print(f"Memory usage: {df.memory_usage(deep=True).sum()} bytes")
print(f"Columns: {list(df.columns)}")
print(f"Index: {df.index.tolist()}")

print("\n" + "="*60)
print("2. DATA EXPLORATION - Know Your Data")
print("="*60)

# Statistical summary
print("Statistical Summary:")
print(df.describe())

# Missing values check
print(f"\nMissing values:\n{df.isnull().sum()}")

# Unique values per column
print(f"\nUnique values per column:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

# Value counts for categorical data
print(f"\nDepartment distribution:")
print(df['department'].value_counts())

print("\n" + "="*60)
print("3. INDEXING & SELECTION - Data Access Patterns")
print("="*60)

# Different ways to select data
print("Column selection:")
print(f"Single column: df['age'] → {type(df['age'])}")
print(f"Multiple columns: df[['name', 'age']] → {type(df[['name', 'age']])}")

# Row selection
print(f"\nRow selection:")
print(f"By index: df.iloc[0] → First row")
print(f"By label: df.loc[0] → Row with index 0")
print(f"Slice: df.iloc[1:3] → Rows 1 and 2")

# Boolean indexing (CRITICAL for ML)
high_salary = df[df['salary'] > 60000]
print(f"\nBoolean indexing - High salary employees:")
print(high_salary[['name', 'salary']])

# Complex conditions
complex_filter = df[(df['age'] > 30) & (df['department'] == 'Engineering')]
print(f"\nComplex filter - Senior Engineers:")
print(complex_filter[['name', 'age', 'department']])

print("\n" + "="*60)
print("4. DATA CLEANING - Preparing for ML")
print("="*60)

# Create data with missing values for demonstration
df_dirty = df.copy()
df_dirty.loc[1, 'salary'] = np.nan
df_dirty.loc[3, 'age'] = np.nan
df_dirty.loc[4, 'experience'] = np.nan

print("Data with missing values:")
print(df_dirty.isnull().sum())

# Handling missing values
print(f"\nMissing value strategies:")

# Strategy 1: Drop rows with any missing values
df_dropped = df_dirty.dropna()
print(f"After dropna(): {df_dropped.shape} (was {df_dirty.shape})")

# Strategy 2: Fill with mean/median/mode
df_filled = df_dirty.copy()
df_filled['salary'].fillna(df_filled['salary'].mean(), inplace=True)
df_filled['age'].fillna(df_filled['age'].median(), inplace=True)
df_filled['experience'].fillna(df_filled['experience'].mode()[0], inplace=True)

print(f"After filling: {df_filled.isnull().sum().sum()} missing values")

# Strategy 3: Forward/backward fill
df_ffill = df_dirty.fillna(method='ffill')  # Forward fill
print(f"Forward fill result: {df_ffill.isnull().sum().sum()} missing values")

print("\n" + "="*60)
print("5. FEATURE ENGINEERING - Creating New Features")
print("="*60)

# Creating new features
df_features = df.copy()

# Numerical transformations
df_features['salary_per_year_exp'] = df_features['salary'] / df_features['experience']
df_features['age_group'] = pd.cut(df_features['age'], bins=[0, 30, 40, 100], labels=['Young', 'Middle', 'Senior'])
df_features['is_high_earner'] = (df_features['salary'] > df_features['salary'].median()).astype(int)

# String operations
df_features['name_length'] = df_features['name'].str.len()
df_features['dept_short'] = df_features['department'].str[:3].str.upper()

print("New features created:")
print(df_features[['name', 'salary_per_year_exp', 'age_group', 'is_high_earner', 'name_length', 'dept_short']])

print("\n" + "="*60)
print("6. GROUPBY OPERATIONS - Aggregation & Analysis")
print("="*60)

# GroupBy operations (essential for feature engineering)
print("Group by department:")
dept_stats = df.groupby('department').agg({
    'salary': ['mean', 'std', 'count'],
    'age': 'mean',
    'experience': 'sum'
})
print(dept_stats)

# Multiple grouping
print(f"\nGroup by department and age group:")
df_temp = df.copy()
df_temp['age_group'] = pd.cut(df_temp['age'], bins=[0, 30, 35, 100], labels=['Young', 'Mid', 'Senior'])
multi_group = df_temp.groupby(['department', 'age_group'])['salary'].mean()
print(multi_group)

# Transform operations (add group statistics to original data)
df['dept_avg_salary'] = df.groupby('department')['salary'].transform('mean')
df['salary_vs_dept_avg'] = df['salary'] - df['dept_avg_salary']
print(f"\nSalary vs department average:")
print(df[['name', 'department', 'salary', 'dept_avg_salary', 'salary_vs_dept_avg']])

print("\n" + "="*60)
print("7. DATA TRANSFORMATION - Preparing for ML Models")
print("="*60)

# One-hot encoding for categorical variables
df_encoded = pd.get_dummies(df, columns=['department'], prefix='dept')
print("One-hot encoded departments:")
print(df_encoded.columns.tolist())
print(df_encoded.head())

# Label encoding (alternative approach)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_label = df.copy()
df_label['department_encoded'] = le.fit_transform(df_label['department'])
print(f"\nLabel encoded departments:")
print(df_label[['department', 'department_encoded']].drop_duplicates())

# Scaling numerical features
from sklearn.preprocessing import StandardScaler, MinMaxScaler

numerical_cols = ['age', 'salary', 'experience']

# Standard scaling (z-score normalization)
scaler_std = StandardScaler()
df_scaled_std = df.copy()
df_scaled_std[numerical_cols] = scaler_std.fit_transform(df[numerical_cols])

# Min-Max scaling
scaler_minmax = MinMaxScaler()
df_scaled_minmax = df.copy()
df_scaled_minmax[numerical_cols] = scaler_minmax.fit_transform(df[numerical_cols])

print(f"\nOriginal vs Scaled data:")
print("Original:")
print(df[numerical_cols].describe())
print("\nStandard Scaled (mean=0, std=1):")
print(df_scaled_std[numerical_cols].describe())
print("\nMin-Max Scaled (range 0-1):")
print(df_scaled_minmax[numerical_cols].describe())

print("\n" + "="*60)
print("8. TIME SERIES OPERATIONS - Temporal Data")
print("="*60)

# Create time series data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
ts_data = pd.DataFrame({
    'date': dates,
    'sales': np.random.randn(100).cumsum() + 100,
    'temperature': 20 + 10 * np.sin(np.arange(100) * 2 * np.pi / 365) + np.random.randn(100)
})

# Set date as index
ts_data.set_index('date', inplace=True)

print("Time series data:")
print(ts_data.head())

# Time-based operations
print(f"\nTime series operations:")
print(f"Resample to weekly mean: {ts_data.resample('W').mean().shape}")
print(f"Rolling 7-day average: {ts_data.rolling(7).mean().shape}")
print(f"Month-over-month change: {ts_data.pct_change(periods=30).shape}")

# Extract time features
ts_data['year'] = ts_data.index.year
ts_data['month'] = ts_data.index.month
ts_data['day_of_week'] = ts_data.index.dayofweek
ts_data['is_weekend'] = (ts_data.index.dayofweek >= 5).astype(int)

print(f"\nTime features extracted:")
print(ts_data[['sales', 'year', 'month', 'day_of_week', 'is_weekend']].head())

print("\n" + "="*60)
print("9. PERFORMANCE OPTIMIZATION - Large Data Handling")
print("="*60)

# Memory optimization
print("Memory optimization techniques:")

# Optimize data types
df_optimized = df.copy()
print(f"Original memory usage: {df.memory_usage(deep=True).sum()} bytes")

# Convert to more efficient types
df_optimized['age'] = df_optimized['age'].astype('int8')  # Age won't exceed 127
df_optimized['experience'] = df_optimized['experience'].astype('int8')
df_optimized['department'] = df_optimized['department'].astype('category')  # Categorical

print(f"Optimized memory usage: {df_optimized.memory_usage(deep=True).sum()} bytes")
print(f"Memory reduction: {(1 - df_optimized.memory_usage(deep=True).sum() / df.memory_usage(deep=True).sum()) * 100:.1f}%")

# Chunked processing for large files
print(f"\nChunked processing example:")
print("# For large CSV files:")
print("# chunk_size = 10000")
print("# for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):")
print("#     process_chunk(chunk)")

print("\n" + "="*60)
print("10. ML PIPELINE INTEGRATION - Ready for Modeling")
print("="*60)

# Complete preprocessing pipeline
def preprocess_for_ml(df):
    """Complete preprocessing pipeline for ML"""
    df_processed = df.copy()
    
    # Handle missing values
    numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    
    # Fill numerical missing values with median
    for col in numerical_cols:
        df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    # Fill categorical missing values with mode
    for col in categorical_cols:
        df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    
    # Encode categorical variables
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
    
    # Scale numerical features
    scaler = StandardScaler()
    df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
    
    return df_processed, scaler

# Apply preprocessing
X_processed, fitted_scaler = preprocess_for_ml(df.drop('salary', axis=1))  # Features
y = df['salary']  # Target

print("Preprocessed features for ML:")
print(f"Feature matrix shape: {X_processed.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature columns: {X_processed.columns.tolist()}")

print("\n" + "="*60)
print("PANDAS CHEAT SHEET FOR ML")
print("="*60)

cheat_sheet = {
    "Data Loading": "pd.read_csv('file.csv')",
    "Quick Info": "df.info(), df.describe(), df.shape",
    "Missing Values": "df.isnull().sum(), df.dropna(), df.fillna()",
    "Selection": "df['col'], df[['col1', 'col2']], df[df['col'] > 5]",
    "Grouping": "df.groupby('col').agg({'col2': 'mean'})",
    "Encoding": "pd.get_dummies(df, columns=['cat_col'])",
    "Scaling": "StandardScaler().fit_transform(df[num_cols])",
    "New Features": "df['new_col'] = df['col1'] / df['col2']",
    "Time Series": "df.resample('D').mean(), df.rolling(7).mean()",
    "Memory Opt": "df['col'].astype('category')"
}

for operation, code in cheat_sheet.items():
    print(f"{operation:.<15} {code}")

print("\n" + "="*60)
print("🎯 PANDAS MASTERY CHECKLIST:")
print("✅ Data loading and exploration")
print("✅ Missing value handling strategies")
print("✅ Feature engineering techniques")
print("✅ Categorical encoding methods")
print("✅ Data scaling and normalization")
print("✅ GroupBy operations for insights")
print("✅ Time series feature extraction")
print("✅ Memory optimization for large data")
print("✅ ML pipeline integration")
print("="*60)