import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

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
        loss=tf.keras.losses.MeanSquaredError()
    )
    
    model.summary()
    return model

def load_lstm_model(model_path='lstm_transaction_model.h5', scaler_path='transaction_scaler.joblib', encoders_path='categorical_encoders.joblib'):
    print(f"Loading model from {model_path}...")
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return None
    if not os.path.exists(scaler_path):
        print(f"Error: Scaler file {scaler_path} not found.")
        return None
    if not os.path.exists(encoders_path):
        print(f"Error: Encoders file {encoders_path} not found.")
        return None

    custom_objects = {'mse': tf.keras.losses.MeanSquaredError()}
    
    # Load the model with custom objects
    lstm_model = load_model(model_path, custom_objects=custom_objects)
    scaler = joblib.load(scaler_path)
    encoders = joblib.load(encoders_path)

    return lstm_model, scaler, encoders

def train_lstm_model(model, X_train, y_train, X_test, y_test, fast_mode=False):
    """
    Train the LSTM model
    
    Args:
        model: Compiled LSTM model
        X_train, y_train: Training data
        X_test, y_test: Validation data
        fast_mode: If True, use settings optimized for faster training
        
    Returns:
        Trained LSTM model
    """
    print("Training LSTM model...")
    
    # Determine batch size based on dataset size
    # For smaller datasets, smaller batch sizes often work better
    num_samples = X_train.shape[0]
    
    if fast_mode:
        # Fast training settings
        epochs = 20
        batch_size = min(32, max(8, num_samples // 100))  # Adaptive batch size
        patience = 5
    else:
        # Full training settings
        epochs = 50
        batch_size = 64
        patience = 10
    
    print(f"Training with batch_size={batch_size}, epochs={epochs}, patience={patience}")
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
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