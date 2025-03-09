import pandas as pd
import kagglehub
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data():
    """
    Load credit card transaction data and perform preprocessing
    
    Returns:
        Processed dataframe and encoders for categorical features
    """
    print("Loading and preprocessing data...")
    
    path = kagglehub.dataset_download("kartik2112/fraud-detection")
    
    # Load the dataset
    df = pd.read_csv(path + "/fraudTrain.csv")
    
    # Check fraud distribution
    print(f"Class distribution:\n{df['is_fraud'].value_counts()}")
    print(f"Fraud percentage: {df['is_fraud'].mean() * 100:.2f}%")
    
    # Convert transaction date to datetime and extract features
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day'] = df['trans_date_trans_time'].dt.day
    df['month'] = df['trans_date_trans_time'].dt.month
    df['dayofweek'] = df['trans_date_trans_time'].dt.dayofweek
    
    # Convert date of birth to age
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = (datetime.now() - df['dob']).dt.days // 365
    
    # Encode categorical features
    categorical_cols = ['merchant', 'category', 'gender', 'city', 'state', 'job']
    encoders = {}
    
    for col in categorical_cols:
        encoder = LabelEncoder()
        df[f'{col}_encoded'] = encoder.fit_transform(df[col])
        encoders[col] = encoder
    
    # Sort by credit card number and transaction time
    df = df.sort_values(['cc_num', 'trans_date_trans_time'])
    
    return df, encoders