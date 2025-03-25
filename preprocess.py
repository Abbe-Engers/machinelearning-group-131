import pandas as pd
import kagglehub
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(sample_size=1.0):
    print("Loading and preprocessing data...")
    
    path = kagglehub.dataset_download("kartik2112/fraud-detection")
    
    df = pd.read_csv(path + "/fraudTrain.csv")
    
    print(f"Class distribution:\n{df['is_fraud'].value_counts()}")
    print(f"Fraud percentage: {df['is_fraud'].mean() * 100:.2f}%")
    
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day'] = df['trans_date_trans_time'].dt.day
    df['month'] = df['trans_date_trans_time'].dt.month
    df['dayofweek'] = df['trans_date_trans_time'].dt.dayofweek
    df = df[df['is_fraud'] == 0]
    
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = (datetime.now() - df['dob']).dt.days // 365
    
    categorical_cols = ['merchant', 'category', 'gender', 'city', 'state', 'job']
    
    for col in categorical_cols:
        encoder = LabelEncoder()
        df[f'{col}_encoded'] = encoder.fit_transform(df[col])
    
    df = df.sort_values(['cc_num', 'trans_date_trans_time'])

    if sample_size < 1.0:
        unique_cc_nums = df['cc_num'].unique()

        cc_nums_amount = int(len(unique_cc_nums) * sample_size)
        sampled_cc_nums = unique_cc_nums[:cc_nums_amount]
        
        df = df[df['cc_num'].isin(sampled_cc_nums)]
        print(f"Sampled {cc_nums_amount} users ({sample_size*100:.1f}% of original dataset)")
        print(f"Sampled dataset shape: {df.shape}")
    
    return df