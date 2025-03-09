import numpy as np
import tensorflow as tf
from model import load_model
import joblib
import argparse

from model import create_lstm_model, prepare_all_sequences, train_lstm_model
from predict import load_model_and_predict_all, predict_next_transaction
from preprocess import load_and_preprocess_data

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def main(sample_size=1.0, fast_mode=False, load_pretrained=False):
    """
    Main function to load data, train model and make predictions
    
    Args:
        sample_size: Float between 0 and 1 indicating the fraction of data to use (default: 1.0 = full dataset)
        fast_mode: If True, use settings optimized for faster training
        load_pretrained: If True, load a pre-trained model instead of training a new one
    """
    if load_pretrained:
        # Load pre-trained model and predict for all credit card numbers
        predictions = load_model_and_predict_all()
        if predictions is not None:
            print("\nSample of predictions:")
            print(predictions.head())
        return
    
    # Load and preprocess data
    df, encoders = load_and_preprocess_data()
    
    # Sample a subset of the data if sample_size < 1
    if sample_size < 1.0:
        # Get unique credit card numbers
        unique_cc_nums = df['cc_num'].unique()
        
        # Sample a subset of credit card numbers
        sampled_cc_nums = np.random.choice(
            unique_cc_nums, 
            size=int(len(unique_cc_nums) * sample_size), 
            replace=False
        )
        
        # Filter the dataframe to only include sampled users
        df = df[df['cc_num'].isin(sampled_cc_nums)]
        print(f"Sampled {len(sampled_cc_nums)} users ({sample_size*100:.1f}% of original dataset)")
        print(f"Sampled dataset shape: {df.shape}")
    
    # Prepare sequences
    sequence_length = 10
    print(f"Preparing sequences with length {sequence_length}...")
    X_train, X_test, y_train, y_test, scaler = prepare_all_sequences(df, sequence_length)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Target data shape: {y_train.shape}")
    
    # Create and train LSTM model
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_shape = y_train.shape[1]
    
    lstm_model = create_lstm_model(input_shape, output_shape)
    lstm_model = train_lstm_model(lstm_model, X_train, y_train, X_test, y_test, fast_mode=fast_mode)
    
    # Save the model and scaler
    lstm_model.save('lstm_transaction_model.h5')
    joblib.dump(scaler, 'transaction_scaler.joblib')
    joblib.dump(encoders, 'categorical_encoders.joblib')
    
    print("Model and preprocessing objects saved.")
    
    # Example: Predict next transaction for a sample user
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Credit Card Transaction Prediction')
    parser.add_argument('--sample_size', type=float, default=0.1, help='Fraction of data to use (0.01-1.0)')
    parser.add_argument('--fast_mode', action='store_true', help='Use faster training settings')
    parser.add_argument('--load_model', action='store_true', help='Load pre-trained model instead of training')
    parser.add_argument('--model_path', type=str, default='lstm_transaction_model.h5', help='Path to pre-trained model')
    
    args = parser.parse_args()
    
    if args.load_model:
        print(f"Loading pre-trained model from {args.model_path}")
        main(load_pretrained=True)
    else:
        print(f"Training on {args.sample_size*100:.1f}% of the original dataset with fast_mode={args.fast_mode}")
        main(sample_size=args.sample_size, fast_mode=args.fast_mode)