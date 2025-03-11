import numpy as np
import tensorflow as tf
import joblib
from model import create_lstm_model, prepare_all_sequences, train_lstm_model
from predict import predict_next_transaction
from preprocess import load_and_preprocess_data

np.random.seed(42)
tf.random.set_seed(42)

def main(sample_size=1.0, fast_mode=False):    
    df, encoders = load_and_preprocess_data()
    
    if sample_size < 1.0:
        unique_cc_nums = df['cc_num'].unique()
        
        sampled_cc_nums = np.random.choice(
            unique_cc_nums, 
            size=int(len(unique_cc_nums) * sample_size), 
            replace=False
        )
        
        df = df[df['cc_num'].isin(sampled_cc_nums)]
        print(f"Sampled {len(sampled_cc_nums)} users ({sample_size*100:.1f}% of original dataset)")
        print(f"Sampled dataset shape: {df.shape}")
    
    sequence_length = 10
    print(f"Preparing sequences with length {sequence_length}...")
    X_train, X_test, y_train, y_test, scaler = prepare_all_sequences(df, sequence_length)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Target data shape: {y_train.shape}")
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_shape = y_train.shape[1]
    
    lstm_model = create_lstm_model(input_shape, output_shape)
    lstm_model = train_lstm_model(lstm_model, X_train, y_train, X_test, y_test, fast_mode=fast_mode)
    
    lstm_model.save('lstm_transaction_model.h5')
    joblib.dump(scaler, 'transaction_scaler.joblib')
    joblib.dump(encoders, 'categorical_encoders.joblib')
    
    print("Model and preprocessing objects saved.")
    
    if len(df) > 0:
        sample_cc_num = df['cc_num'].iloc[0]
        user_transactions = df[df['cc_num'] == sample_cc_num]
        
        if len(user_transactions) > sequence_length:
            next_transaction = predict_next_transaction(
                lstm_model, user_transactions, sequence_length, scaler, df.columns, encoders
            )
            
            print("\nPredicted next transaction:")
            print(next_transaction)

if __name__ == "__main__":    
    sample_size = 0.1
    fast_mode = True
    
    print(f"Training on {sample_size*100:.1f}% of the original dataset with fast_mode={fast_mode}")
    main(sample_size=sample_size, fast_mode=fast_mode)