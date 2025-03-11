import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

# Define the number of bins for continuous features
NUM_BINS = {
    'amt': 50,  # More bins for amount since it's highly variable
    'hour': 24,  # 24 hours in a day
    'day': 31,  # Maximum days in a month
    'month': 12,  # 12 months
    'dayofweek': 7,  # 7 days in a week
    'lat': 30,  # Latitude bins
    'long': 30,  # Longitude bins
    'merch_lat': 30,  # Merchant latitude bins
    'merch_long': 30  # Merchant longitude bins
}

class TransactionFeatureProcessor:
    def __init__(self):
        self.continuous_features = ['amt', 'lat', 'long', 'merch_lat', 'merch_long']
        self.discrete_features = ['hour', 'day', 'month', 'dayofweek']
        self.categorical_features = ['merchant_encoded', 'category_encoded']
        self.target = 'is_fraud'
        
        self.discretizers = {}
        self.max_values = {}
        
    def fit(self, data):
        # Initialize discretizers for continuous features
        for feature in self.continuous_features:
            self.discretizers[feature] = KBinsDiscretizer(
                n_bins=NUM_BINS[feature], 
                encode='onehot', 
                strategy='quantile'
            )
            feature_data = data[feature].values.reshape(-1, 1)
            self.discretizers[feature].fit(feature_data)
        
        # Store max values for discrete features
        for feature in self.discrete_features:
            self.max_values[feature] = int(data[feature].max()) + 1
            
        # Store max values for categorical features
        for feature in self.categorical_features:
            self.max_values[feature] = int(data[feature].max()) + 1
    
    def transform_feature(self, data, feature):
        if feature in self.continuous_features:
            return self.discretizers[feature].transform(
                data[feature].values.reshape(-1, 1)
            ).toarray()
        elif feature in self.discrete_features:
            return tf.keras.utils.to_categorical(
                data[feature], 
                num_classes=self.max_values[feature]
            )
        elif feature in self.categorical_features:
            return tf.keras.utils.to_categorical(
                data[feature], 
                num_classes=self.max_values[feature]
            )
        else:
            return tf.keras.utils.to_categorical(
                data[feature], 
                num_classes=2
            )

def create_sequences(user_df, sequence_length=10, processor=None):
    features = [
        'amt', 'hour', 'day', 'month', 'dayofweek', 
        'merchant_encoded', 'category_encoded', 'lat', 'long',
        'merch_lat', 'merch_long', 'is_fraud'
    ]
    
    if len(user_df) <= sequence_length:
        return None, None
    
    X, y = [], []
    for i in range(len(user_df) - sequence_length):
        sequence = user_df[features].iloc[i:i+sequence_length]
        target = user_df[features].iloc[i+sequence_length]
        
        X.append(sequence.values)
        
        # Transform each feature in the target to its categorical representation
        target_transformed = {
            feature: processor.transform_feature(
                target.to_frame().T, 
                feature
            ) for feature in features
        }
        y.append(target_transformed)
    
    return np.array(X), y

def prepare_all_sequences(df, sequence_length=10):
    # Initialize and fit the feature processor
    processor = TransactionFeatureProcessor()
    processor.fit(df)
    
    all_X, all_y = [], []
    for cc_num, group in df.groupby('cc_num'):
        X, y = create_sequences(group, sequence_length, processor)
        if X is not None and y is not None:
            all_X.append(X)
            all_y.extend(y)
    
    X = np.vstack(all_X)
    
    # Normalize the input sequences
    scaler = StandardScaler()
    X_shape = X.shape
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.fit_transform(X_reshaped)
    X = X_scaled.reshape(X_shape)
    
    # Split the data
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train = [all_y[i] for i in train_idx]
    y_test = [all_y[i] for i in test_idx]
    
    return X_train, X_test, y_train, y_test, scaler, processor

def create_lstm_model(input_shape, processor):
    # Create input layer
    input_layer = Input(shape=input_shape)
    
    # LSTM layers
    x = LSTM(128, return_sequences=True)(input_layer)
    x = Dropout(0.3)(x)
    x = LSTM(256, return_sequences=True)(x)
    x = Dropout(0.4)(x)
    x = LSTM(128)(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    
    # Create output layers for each feature
    outputs = {}
    
    # Continuous features
    for feature in processor.continuous_features:
        outputs[feature] = Dense(
            NUM_BINS[feature], 
            activation='softmax', 
            name=feature
        )(x)
    
    # Discrete features
    for feature in processor.discrete_features:
        outputs[feature] = Dense(
            processor.max_values[feature], 
            activation='softmax', 
            name=feature
        )(x)
    
    # Categorical features
    for feature in processor.categorical_features:
        outputs[feature] = Dense(
            processor.max_values[feature], 
            activation='softmax', 
            name=feature
        )(x)
    
    # Fraud classification
    outputs['is_fraud'] = Dense(2, activation='softmax', name='is_fraud')(x)
    
    # Create and compile model
    model = Model(inputs=input_layer, outputs=outputs)
    
    # Define loss and metrics for each output
    losses = {name: 'categorical_crossentropy' for name in outputs.keys()}
    metrics = {name: 'accuracy' for name in outputs.keys()}
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=losses,
        metrics=metrics
    )
    
    model.summary()
    return model

def train_lstm_model(model, X_train, y_train, X_test, y_test, fast_mode=False):
    print("Training LSTM model...")
    num_samples = X_train.shape[0]
    
    if fast_mode:
        epochs = 1
        # epochs = 20
        batch_size = min(32, max(8, num_samples // 100))
        patience = 5
    else:
        epochs = 50
        batch_size = 64
        patience = 10
    
    print(f"Training with batch_size={batch_size}, epochs={epochs}, patience={patience}")
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )
    
    # Convert y_train and y_test to the format expected by the model
    # Fix the shape mismatch by ensuring each feature array has the correct dimensions
    y_train_dict = {}
    for feature in y_train[0].keys():
        stacked_data = np.stack([y[feature] for y in y_train])
        # Check if the stacked data has 3 dimensions and needs to be reshaped
        if stacked_data.ndim == 3:
            # Reshape from (samples, 1, classes) to (samples, classes)
            stacked_data = stacked_data.reshape(stacked_data.shape[0], stacked_data.shape[2])
        y_train_dict[feature] = stacked_data
    
    y_test_dict = {}
    for feature in y_test[0].keys():
        stacked_data = np.stack([y[feature] for y in y_test])
        # Check if the stacked data has 3 dimensions and needs to be reshaped
        if stacked_data.ndim == 3:
            # Reshape from (samples, 1, classes) to (samples, classes)
            stacked_data = stacked_data.reshape(stacked_data.shape[0], stacked_data.shape[2])
        y_test_dict[feature] = stacked_data
    
    history = model.fit(
        X_train,
        y_train_dict,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test_dict),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 1, 1)
    for output in model.output_names:
        plt.plot(history.history[f'{output}_loss'], 
                label=f'{output} (train)')
        plt.plot(history.history[f'val_{output}_loss'], 
                label=f'{output} (val)')
    plt.title('Model Loss by Feature')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot accuracy
    plt.subplot(2, 1, 2)
    for output in model.output_names:
        plt.plot(history.history[f'{output}_accuracy'], 
                label=f'{output} (train)')
        plt.plot(history.history[f'val_{output}_accuracy'], 
                label=f'{output} (val)')
    plt.title('Model Accuracy by Feature')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('models/lstm_training_history.png', bbox_inches='tight')
    
    return model

def load_lstm_model(model_path='models/lstm_transaction_model.h5', 
                   scaler_path='models/transaction_scaler.joblib', 
                   processor_path='models/transaction_processor.joblib'):
    print(f"Loading model from {model_path}...")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return None
    if not os.path.exists(scaler_path):
        print(f"Error: Scaler file {scaler_path} not found.")
        return None
    if not os.path.exists(processor_path):
        print(f"Error: Processor file {processor_path} not found.")
        return None

    custom_objects = {'categorical_crossentropy': tf.keras.losses.CategoricalCrossentropy()}
    
    lstm_model = load_model(model_path, custom_objects=custom_objects)
    scaler = joblib.load(scaler_path)
    processor = joblib.load(processor_path)

    return lstm_model, scaler, processor