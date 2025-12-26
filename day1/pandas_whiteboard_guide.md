# Pandas Whiteboard Visual Guide - ML Focus

## 1. DataFrame Structure Visualization

```
DATAFRAME ANATOMY:

        Columns (Features)
         ↓  ↓  ↓  ↓
    ┌─────┬────┬────┬────┐
  0 │Alice│ 25 │ 50k│Eng │ ← Row (Sample/Record)
    ├─────┼────┼────┼────┤
  1 │ Bob │ 30 │ 60k│Mkt │
    ├─────┼────┼────┼────┤
  2 │Charlie│35│ 70k│Eng │
    └─────┴────┴────┴────┘
    ↑
   Index

Key Concepts:
- Each row = one data sample/observation
- Each column = one feature/variable
- Index = unique identifier for rows
- Values = actual data points
```

## 2. Boolean Indexing (Critical for ML)

```
BOOLEAN INDEXING FLOW:

Original DataFrame:
┌─────┬────┬────────┐
│Name │Age │ Salary │
├─────┼────┼────────┤
│Alice│ 25 │  50000 │
│ Bob │ 30 │  60000 │
│Charlie│35│  70000 │
│Diana│ 28 │  55000 │
└─────┴────┴────────┘

Step 1: Create Boolean Mask
df['Salary'] > 55000
↓
[False, True, True, False]

Step 2: Apply Mask
┌─────┬────┬────────┐
│ Bob │ 30 │  60000 │ ← Only True rows
│Charlie│35│  70000 │
└─────┴────┴────────┘

Use Cases:
- Filter training data
- Remove outliers
- Create train/test splits
- Feature selection
```

## 3. GroupBy Operations (Feature Engineering)

```
GROUPBY VISUALIZATION:

Original Data:
┌─────┬──────┬────────┐
│Name │ Dept │ Salary │
├─────┼──────┼────────┤
│Alice│ Eng  │  50000 │
│ Bob │ Mkt  │  60000 │
│Charlie│Eng │  70000 │
│Diana│ HR   │  55000 │
│ Eve │ Mkt  │  65000 │
└─────┴──────┴────────┘

GroupBy Department:
┌─────────────────────┐
│ Engineering Group   │
│ ┌─────┬────────┐   │
│ │Alice│  50000 │   │
│ │Charlie│70000 │   │
│ └─────┴────────┘   │
└─────────────────────┘
┌─────────────────────┐
│ Marketing Group     │
│ ┌─────┬────────┐   │
│ │ Bob │  60000 │   │
│ │ Eve │  65000 │   │
│ └─────┴────────┘   │
└─────────────────────┘

Aggregation Results:
┌──────┬─────────┬─────┬───────┐
│ Dept │ Mean    │Count│ Std   │
├──────┼─────────┼─────┼───────┤
│ Eng  │ 60000   │  2  │ 14142 │
│ Mkt  │ 62500   │  2  │  3536 │
│ HR   │ 55000   │  1  │   NaN │
└──────┴─────────┴─────┴───────┘
```

## 4. One-Hot Encoding Transformation

```
ONE-HOT ENCODING PROCESS:

Before Encoding:
┌─────┬──────────┐
│Name │Department│
├─────┼──────────┤
│Alice│    Eng   │
│ Bob │    Mkt   │
│Charlie│   Eng   │
│Diana│    HR    │
└─────┴──────────┘

After pd.get_dummies():
┌─────┬────────┬────────┬───────┐
│Name │Dept_Eng│Dept_HR │Dept_Mkt│
├─────┼────────┼────────┼───────┤
│Alice│   1    │   0    │   0   │
│ Bob │   0    │   0    │   1   │
│Charlie│  1    │   0    │   0   │
│Diana│   0    │   1    │   0   │
└─────┴────────┴────────┴───────┘

Why One-Hot Encoding?
- ML algorithms need numerical input
- Avoids ordinal assumptions (Eng ≠ 1, Mkt ≠ 2)
- Each category becomes a binary feature
- Prevents model from assuming order/distance
```

## 5. Missing Value Strategies

```
MISSING VALUE HANDLING:

Original Data with NaN:
┌─────┬────┬────────┐
│Name │Age │ Salary │
├─────┼────┼────────┤
│Alice│ 25 │  50000 │
│ Bob │NaN │  60000 │
│Charlie│35│   NaN  │
│Diana│ 28 │  55000 │
└─────┴────┴────────┘

Strategy 1: Drop Rows (dropna())
┌─────┬────┬────────┐
│Alice│ 25 │  50000 │
│Diana│ 28 │  55000 │
└─────┴────┴────────┘
Pros: Clean data | Cons: Data loss

Strategy 2: Fill with Mean/Median
┌─────┬────┬────────┐
│Alice│ 25 │  50000 │
│ Bob │ 29 │  60000 │ ← Age filled with median
│Charlie│35│  55000 │ ← Salary filled with mean
│Diana│ 28 │  55000 │
└─────┴────┴────────┘
Pros: Keep all data | Cons: May introduce bias

Strategy 3: Forward Fill (ffill())
Uses previous valid value
Strategy 4: Interpolation
Estimates based on surrounding values
```

## 6. Feature Engineering Pipeline

```
FEATURE ENGINEERING FLOW:

Raw Data → Clean → Transform → Encode → Scale → ML Ready

Step 1: Raw Data
┌─────┬────┬────────┬──────┐
│Name │Age │ Salary │ Dept │
├─────┼────┼────────┼──────┤
│Alice│ 25 │  50000 │ Eng  │
│ Bob │ 30 │  60000 │ Mkt  │
└─────┴────┴────────┴──────┘

Step 2: Create New Features
┌─────┬────┬────────┬──────┬─────────┬──────────┐
│Name │Age │ Salary │ Dept │Age_Group│Sal_Per_Yr│
├─────┼────┼────────┼──────┼─────────┼──────────┤
│Alice│ 25 │  50000 │ Eng  │ Young   │  25000   │
│ Bob │ 30 │  60000 │ Mkt  │ Adult   │  20000   │
└─────┴────┴────────┴──────┴─────────┴──────────┘

Step 3: Encode Categoricals
┌─────┬────┬────────┬────────┬────────┬─────────┬──────────┐
│Name │Age │ Salary │Dept_Eng│Dept_Mkt│Age_Group│Sal_Per_Yr│
├─────┼────┼────────┼────────┼────────┼─────────┼──────────┤
│Alice│ 25 │  50000 │   1    │   0    │ Young   │  25000   │
│ Bob │ 30 │  60000 │   0    │   1    │ Adult   │  20000   │
└─────┴────┴────────┴────────┴────────┴─────────┴──────────┘

Step 4: Scale Numerical Features
┌─────┬──────┬────────┬────────┬────────┬─────────┬──────────┐
│Name │ Age  │ Salary │Dept_Eng│Dept_Mkt│Age_Group│Sal_Per_Yr│
├─────┼──────┼────────┼────────┼────────┼─────────┼──────────┤
│Alice│-0.71 │ -0.71  │   1    │   0    │ Young   │  0.71    │
│ Bob │ 0.71 │  0.71  │   0    │   1    │ Adult   │ -0.71    │
└─────┴──────┴────────┴────────┴────────┴─────────┴──────────┘
```

## 7. Time Series Feature Extraction

```
TIME SERIES DECOMPOSITION:

Original Time Series:
Date        Sales
2023-01-01   100
2023-01-02   105
2023-01-03   98
...

Extract Time Features:
┌────────────┬─────┬────┬─────┬────┬─────────┬───────────┐
│    Date    │Sales│Year│Month│Day │DayOfWeek│IsWeekend  │
├────────────┼─────┼────┼─────┼────┼─────────┼───────────┤
│2023-01-01  │ 100 │2023│  1  │ 1  │    6    │     1     │
│2023-01-02  │ 105 │2023│  1  │ 2  │    0    │     0     │
│2023-01-03  │  98 │2023│  1  │ 3  │    1    │     0     │
└────────────┴─────┴────┴─────┴────┴─────────┴───────────┘

Rolling Statistics:
┌────────────┬─────┬──────────┬──────────┬──────────┐
│    Date    │Sales│7Day_Mean │7Day_Std  │7Day_Trend│
├────────────┼─────┼──────────┼──────────┼──────────┤
│2023-01-07  │ 110 │  102.3   │   4.2    │   +2.1   │
│2023-01-08  │ 115 │  104.1   │   5.1    │   +1.8   │
└────────────┴─────┴──────────┴──────────┴──────────┘

Seasonal Decomposition:
Sales = Trend + Seasonal + Residual
  110 =   105  +    8     +   -3
```

## 8. Memory Optimization Strategies

```
MEMORY OPTIMIZATION:

Before Optimization:
┌──────────┬─────────┬──────────────┐
│ Column   │  Type   │ Memory (MB)  │
├──────────┼─────────┼──────────────┤
│ ID       │ int64   │     8.0      │
│ Age      │ int64   │     8.0      │
│ Category │ object  │    50.0      │
│ Score    │ float64 │     8.0      │
└──────────┴─────────┴──────────────┘
Total: 74 MB

After Optimization:
┌──────────┬─────────┬──────────────┐
│ Column   │  Type   │ Memory (MB)  │
├──────────┼─────────┼──────────────┤
│ ID       │ int32   │     4.0      │
│ Age      │ int8    │     1.0      │
│ Category │category │    10.0      │
│ Score    │ float32 │     4.0      │
└──────────┴─────────┴──────────────┘
Total: 19 MB (74% reduction!)

Optimization Rules:
- int64 → int32/int16/int8 (if range allows)
- float64 → float32 (if precision allows)
- object → category (for repeated strings)
- Use sparse arrays for mostly zeros
```

## Key Whiteboard Drawing Tips for Pandas:

1. **Always show before/after** for transformations
2. **Use tables** to represent DataFrames clearly
3. **Show data flow** with arrows between steps
4. **Highlight missing values** with special notation (NaN, ?)
5. **Use different colors** for different data types
6. **Show dimensions** (rows × columns) for each step
7. **Include memory/performance** implications

## Interview Questions You Can Now Answer:

1. "How do you handle missing values in a dataset?"
2. "Explain the difference between loc and iloc"
3. "How would you encode categorical variables for ML?"
4. "What's the best way to create features from datetime data?"
5. "How do you optimize memory usage for large datasets?"
6. "Explain groupby operations and their use in feature engineering"
7. "How do you prepare a pandas DataFrame for machine learning?"

## Practice Exercises:

1. Draw the one-hot encoding process for a categorical column
2. Show how groupby creates aggregate features
3. Illustrate the missing value handling decision tree
4. Draw a complete feature engineering pipeline
5. Show memory optimization before/after comparison