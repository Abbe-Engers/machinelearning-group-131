import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import kagglehub
from datetime import datetime
import joblib

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

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

def create_sequences(user_df, sequence_length=10):
    """
    Create sequences for LSTM model for a single user
    
    Args:
        user_df: DataFrame containing transactions for a single user
        sequence_length: Number of transactions to use for predicting the next one
        
    Returns:
        X: Input sequences
        y: Target values (next transaction)
    """
    # Select features for sequence
    features = [
        'amt', 'hour', 'day', 'month', 'dayofweek', 
        'merchant_encoded', 'category_encoded', 'lat', 'long',
        'merch_lat', 'merch_long', 'is_fraud'
    ]
    
    # Create sequences
    X, y = [], []
    data = user_df[features].values
    
    if len(data) <= sequence_length:
        return None, None
    
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        
        # For y, we want to predict all features of the next transaction
        y.append(data[i+sequence_length])
    
    return np.array(X), np.array(y)

def prepare_all_sequences(df, sequence_length=10):
    """
    Prepare sequences for all users
    
    Args:
        df: Processed DataFrame
        sequence_length: Length of input sequences
        
    Returns:
        X_train, X_test, y_train, y_test: Train and test sequences
    """
    all_X, all_y = [], []
    
    # Group by credit card number
    for cc_num, group in df.groupby('cc_num'):
        X, y = create_sequences(group, sequence_length)
        
        if X is not None and y is not None:
            all_X.append(X)
            all_y.append(y)
    
    # Combine sequences from all users
    X = np.vstack(all_X)
    y = np.vstack(all_y)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    
    # Reshape to 2D for scaling
    X_train_shape = X_train.shape
    X_test_shape = X_test.shape
    
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)
    
    # Reshape back to 3D
    X_train = X_train_scaled.reshape(X_train_shape)
    X_test = X_test_scaled.reshape(X_test_shape)
    
    # Scale y values
    y_train = scaler.transform(y_train)
    y_test = scaler.transform(y_test)
    
    return X_train, X_test, y_train, y_test, scaler

def create_lstm_model(input_shape, output_shape):
    """
    Create LSTM model for transaction prediction
    
    Args:
        input_shape: Shape of input sequences
        output_shape: Shape of output (number of features to predict)
        
    Returns:
        Compiled LSTM model
    """
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(256, return_sequences=True),
        Dropout(0.4),
        LSTM(128),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(output_shape)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    
    model.summary()
    return model

def train_lstm_model(model, X_train, y_train, X_test, y_test):
    """
    Train the LSTM model
    
    Args:
        model: Compiled LSTM model
        X_train, y_train: Training data
        X_test, y_test: Validation data
        
    Returns:
        Trained LSTM model
    """
    print("Training LSTM model...")
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('lstm_training_history.png')
    
    return model

def predict_next_transaction(model, user_transactions, sequence_length, scaler, df_columns, encoders):
    """
    Predict the next transaction for a user
    
    Args:
        model: Trained LSTM model
        user_transactions: DataFrame containing user's transaction history
        sequence_length: Length of input sequences
        scaler: Fitted StandardScaler
        df_columns: Original DataFrame columns
        encoders: Dictionary of LabelEncoders for categorical features
        
    Returns:
        Predicted next transaction as a DataFrame row
    """
    # Prepare the input sequence
    features = [
        'amt', 'hour', 'day', 'month', 'dayofweek', 
        'merchant_encoded', 'category_encoded', 'lat', 'long',
        'merch_lat', 'merch_long', 'is_fraud'
    ]
    
    # Get the last sequence_length transactions
    last_transactions = user_transactions[features].values[-sequence_length:]
    
    # Reshape and scale
    X = np.array([last_transactions])
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.transform(X_reshaped)
    X = X_scaled.reshape(X.shape)
    
    # Make prediction
    pred_scaled = model.predict(X)[0]
    
    # Inverse transform the prediction
    pred = scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]
    
    # Create a DataFrame with the prediction
    pred_df = pd.DataFrame(columns=features)
    for i, col in enumerate(features):
        pred_df.loc[0, col] = pred[i]
    
    # Convert encoded values back to original categories
    for col in ['merchant', 'category']:
        encoded_col = f'{col}_encoded'
        if encoded_col in pred_df.columns:
            # Find the closest encoded value
            encoded_val = int(round(pred_df[encoded_col].values[0]))
            # Ensure it's within valid range
            encoded_val = max(0, min(encoded_val, len(encoders[col].classes_) - 1))
            # Decode
            pred_df[col] = encoders[col].inverse_transform([encoded_val])[0]
    
    # Round is_fraud to 0 or 1
    pred_df['is_fraud'] = (pred_df['is_fraud'] > 0.5).astype(int)
    
    return pred_df

def main():
    # Load and preprocess data
    df, encoders = load_and_preprocess_data()
    
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
    lstm_model = train_lstm_model(lstm_model, X_train, y_train, X_test, y_test)
    
    # Save the model and scaler
    lstm_model.save('lstm_transaction_model.h5')
    joblib.dump(scaler, 'transaction_scaler.joblib')
    joblib.dump(encoders, 'categorical_encoders.joblib')
    
    print("Model and preprocessing objects saved.")
    
    # Example: Predict next transaction for a sample user
    sample_cc_num = df['cc_num'].iloc[0]
    user_transactions = df[df['cc_num'] == sample_cc_num]
    
    if len(user_transactions) > sequence_length:
        next_transaction = predict_next_transaction(
            lstm_model, user_transactions, sequence_length, scaler, df.columns, encoders
        )
        
        print("\nPredicted next transaction:")
        print(next_transaction)
        
        # Function to predict full transaction details
        def predict_full_transaction(user_cc_num, model, df, sequence_length, scaler, encoders):
            """
            Predict complete next transaction for a given credit card
            
            Args:
                user_cc_num: Credit card number
                model: Trained LSTM model
                df: Original DataFrame
                sequence_length: Length of input sequences
                scaler: Fitted StandardScaler
                encoders: Dictionary of LabelEncoders
                
            Returns:
                Complete predicted transaction as a DataFrame row
            """
            # Get user's transaction history
            user_df = df[df['cc_num'] == user_cc_num].copy()
            
            if len(user_df) <= sequence_length:
                print(f"Not enough transaction history for cc_num {user_cc_num}")
                return None
            
            # Get basic prediction
            pred_basic = predict_next_transaction(
                model, user_df, sequence_length, scaler, df.columns, encoders
            )
            
            # Create a complete transaction record
            full_pred = user_df.iloc[-1:].copy()
            
            # Update with predicted values
            for col in pred_basic.columns:
                if col in full_pred.columns:
                    full_pred[col] = pred_basic[col].values[0]
            
            # Update transaction time (e.g., add 1 day to last transaction)
            full_pred['trans_date_trans_time'] = user_df['trans_date_trans_time'].iloc[-1] + pd.Timedelta(days=1)
            
            # Generate a new transaction number
            full_pred['trans_num'] = f"TX{np.random.randint(1000000000, 9999999999)}"
            
            return full_pred
        
        print("\nExample function for predicting complete transactions created.")

if __name__ == "__main__":
    main()