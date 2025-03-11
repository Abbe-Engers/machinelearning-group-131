import numpy as np
import tensorflow as tf
import joblib
from model import create_lstm_model, prepare_all_sequences, train_lstm_model
from predict import predict_and_analyze
import os
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
    X_train, X_test, y_train, y_test, scaler, processor = prepare_all_sequences(df, sequence_length)
    
    print(f"Training data shape: {X_train.shape}")
    print("Training target shapes:")
    for feature, data in y_train[0].items():
        print(f"  {feature}: {data.shape}")
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    print("\nCreating and training LSTM model...")
    lstm_model = create_lstm_model(input_shape, processor)
    lstm_model = train_lstm_model(lstm_model, X_train, y_train, X_test, y_test, fast_mode=fast_mode)
    
    # Save model and preprocessing objects
    print("\nSaving model and preprocessing objects...")
    os.makedirs('models', exist_ok=True)
    lstm_model.save('models/lstm_transaction_model.h5')
    joblib.dump(scaler, 'models/transaction_scaler.joblib')
    joblib.dump(processor, 'models/transaction_processor.joblib')
    
    print("Model and preprocessing objects saved.")
    
    # Generate sample prediction
    if len(df) > 0:
        print("\nGenerating sample prediction...")
        sample_cc_num = df['cc_num'].iloc[0]
        user_transactions = df[df['cc_num'] == sample_cc_num]
        
        if len(user_transactions) > sequence_length:
            # Get the last transaction for comparison
            last_transaction = user_transactions.iloc[-1].to_dict()
            
            # Predict and analyze
            results = predict_and_analyze(
                lstm_model,
                user_transactions.iloc[:-1],  # Use all but last transaction
                sequence_length,
                processor,
                actual_transaction=last_transaction
            )
            
            # Print prediction results
            print("\nPrediction Analysis:")
            print("-------------------")
            print("Most likely values:")
            for feature, value in results['prediction']['most_likely_values'].items():
                print(f"  {feature}: {value}")
            
            print("\nFraud Probability:", 
                  f"{results['prediction']['fraud_probability']:.2%}")
            
            print("\nInsights:")
            for insight in results['insights']:
                print(f"  {insight}")
            
            if results['anomaly_score'] is not None:
                print(f"\nOverall Anomaly Score: {results['anomaly_score']:.2f}")
                print("\nFeature Anomaly Scores:")
                for feature, score in results['feature_scores'].items():
                    print(f"  {feature}: {score:.2f}")
            
            # Save prediction visualization
            print("\nPrediction distribution plots saved to 'prediction_distributions.png'")

if __name__ == "__main__":    
    sample_size = 0.1
    fast_mode = True
    
    print(f"Training on {sample_size*100:.1f}% of the original dataset with fast_mode={fast_mode}")
    main(sample_size=sample_size, fast_mode=fast_mode)