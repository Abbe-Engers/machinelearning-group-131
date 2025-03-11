import pandas as pd
import numpy as np

from model import load_lstm_model, load_model
from preprocess import load_and_preprocess_data

def predict_next_transaction(model, user_transactions, sequence_length, scaler, df_columns, encoders):
    features = [
        'amt', 'hour', 'day', 'month', 'dayofweek', 
        'merchant_encoded', 'category_encoded', 'lat', 'long',
        'merch_lat', 'merch_long', 'is_fraud'
    ]
    
    last_transactions = user_transactions[features].values[-sequence_length:]
    
    X = np.array([last_transactions])
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.transform(X_reshaped)
    X = X_scaled.reshape(X.shape)
    
    pred_scaled = model.predict(X)[0]
    
    pred = scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]
    
    pred_df = pd.DataFrame(columns=features)
    for i, col in enumerate(features):
        pred_df.loc[0, col] = pred[i]
    
    for col in ['merchant', 'category']:
        encoded_col = f'{col}_encoded'
        if encoded_col in pred_df.columns:
            encoded_val = int(round(pred_df[encoded_col].values[0]))
            encoded_val = max(0, min(encoded_val, len(encoders[col].classes_) - 1))
            pred_df[col] = encoders[col].inverse_transform([encoded_val])[0]
    
    pred_df['is_fraud'] = (pred_df['is_fraud'] > 0.5).astype(int)
    
    return pred_df

def load_model_and_predict_all(model_path='lstm_transaction_model.h5', scaler_path='transaction_scaler.joblib', encoders_path='categorical_encoders.joblib'):
    lstm_model, scaler, encoders = load_lstm_model(model_path, scaler_path, encoders_path)
    
    try:
        df, _ = load_and_preprocess_data()
    except Exception as e:
        print(f"Error loading and preprocessing data: {e}")
        return None
    
    sequence_length = 10
    
    predictions = {}
    
    print(f"Predicting next transactions for {len(df['cc_num'].unique())} credit cards...")
    
    total_cards = len(df['cc_num'].unique())
    processed = 0
    
    for cc_num, user_transactions in df.groupby('cc_num'):
        if len(user_transactions) > sequence_length:
            try:
                next_transaction = predict_next_transaction(
                    lstm_model, user_transactions, sequence_length, scaler, df.columns, encoders
                )
                
                next_transaction['cc_num'] = cc_num
                
                predictions[cc_num] = next_transaction
                
                processed += 1
                if processed % 10 == 0 or processed == total_cards:
                    print(f"Progress: {processed}/{total_cards} credit cards processed")
                
            except Exception as e:
                print(f"Error predicting for credit card {cc_num}: {e}")
    
    if predictions:
        all_predictions = pd.concat(predictions.values(), ignore_index=True)
        
        all_predictions.to_csv('predicted_transactions.csv', index=False)
        print(f"Saved predictions for {len(predictions)} credit cards to 'predicted_transactions.csv'")
        
        return all_predictions
    else:
        print("No predictions were generated.")
        return None