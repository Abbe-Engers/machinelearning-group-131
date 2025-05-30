import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import load_lstm_model
from preprocess import load_and_preprocess_data

def prepare_sequence(transactions, sequence_length, processor):
    """Prepare a sequence of transactions for prediction."""
    if len(transactions) < sequence_length:
        raise ValueError(f"Not enough transactions. Need at least {sequence_length}, got {len(transactions)}")
    
    recent_transactions = transactions.iloc[-sequence_length:]
    sequence = recent_transactions[processor.all_features()].values
    
    return sequence.reshape(1, sequence_length, -1)

def get_feature_bins(processor, feature, prediction_probs):
    """Get the bin edges and probabilities for a feature."""
    if feature in processor.continuous_features:
        # Get bin edges from the discretizer
        bin_edges = processor.discretizers[feature].bin_edges_[0]
        return bin_edges, prediction_probs[feature][0]
    elif feature in processor.discrete_features:
        # For discrete features, bins are just the possible values
        return np.arange(processor.max_values[feature]), prediction_probs[feature][0]
    else:
        return None, prediction_probs[feature][0]

def plot_feature_distribution(processor, feature, prediction_probs, ax=None, historical_data=None, fraud_data=None):
    """Plot the probability distribution for a feature."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    
    bin_edges, probs = get_feature_bins(processor, feature, prediction_probs)
    
    if feature in processor.continuous_features:
        # For continuous features, plot as a continuous distribution
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        if feature == 'amt':
            ax.set_xscale('log')
            ax.bar(centers, probs, width=np.diff(bin_edges), alpha=0.6, label='Predicted')
        else:
            ax.bar(centers, probs, width=np.diff(bin_edges), alpha=0.6, label='Predicted')
        
        # Plot historical distribution if available
        if historical_data is not None:
            hist_values = historical_data[feature].values
            ax.hist(hist_values, bins=bin_edges, density=True, alpha=1, 
                   color='orange', label='Historical', histtype='step')
            
        # Plot fraud distribution if available
        if fraud_data is not None and len(fraud_data) > 0:
            fraud_values = fraud_data[feature].values
            # Scale fraud histogram to match the height of the predicted distribution
            hist, _ = np.histogram(fraud_values, bins=bin_edges, density=True)
            scale_factor = np.max(probs) / np.max(hist) if np.max(hist) > 0 else 1
            ax.hist(fraud_values, bins=bin_edges, density=True, alpha=0.6,
                   color='red', label='Fraud', histtype='step', weights=np.ones_like(fraud_values) * scale_factor)
            
        ax.set_xlabel(f"{feature} value")
    else:
        # For discrete features, plot as a bar chart
        ax.bar(range(len(probs)), probs, alpha=0.6, label='Predicted')
        
        # Plot historical distribution if available
        if historical_data is not None:
            hist_dist = historical_data[feature].value_counts(normalize=True)
            ax.plot(range(len(probs)), [hist_dist.get(i, 0) for i in range(len(probs))], 
                   color='orange', alpha=1, label='Historical', marker='o')
            
        # Plot fraud distribution if available
        if fraud_data is not None and len(fraud_data) > 0:
            fraud_dist = fraud_data[feature].value_counts(normalize=True)
            # Scale fraud distribution to match the height of the predicted distribution
            fraud_probs = [fraud_dist.get(i, 0) for i in range(len(probs))]
            scale_factor = np.max(probs) / np.max(fraud_probs) if np.max(fraud_probs) > 0 else 1
            ax.plot(range(len(probs)), [p * scale_factor for p in fraud_probs],
                   color='red', alpha=0.6, label='Fraud', marker='o')
            
        if feature in processor.discrete_features:
            if feature == 'hour':
                ax.set_xticks(range(24))
            elif feature == 'day':
                ax.set_xticks(range(1, 32))
            elif feature == 'month':
                ax.set_xticks(range(1, 13))
            elif feature == 'dayofweek':
                ax.set_xticks(range(7))
                ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    
    ax.set_ylabel('Probability')
    ax.set_title(f'{feature} Distribution')
    ax.legend()
    return ax

def get_most_likely_value(processor, feature, prediction_probs):
    """Get the most likely value for a feature."""
    bin_edges, probs = get_feature_bins(processor, feature, prediction_probs)
    
    if feature in processor.continuous_features:
        most_likely_bin = np.argmax(probs)
        lower_bound = bin_edges[most_likely_bin]
        upper_bound = bin_edges[most_likely_bin + 1]
        center = (lower_bound + upper_bound) / 2
        return {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'avg': center
        }
    else:
        return {
            'avg': np.argmax(probs)
        }

def calculate_anomaly_score(prediction_probs, actual_values, processor):
    """Calculate an anomaly score based on the probability distributions."""
    feature_scores = {}
    feature_probabilities = {}
    
    for feature in processor.continuous_features + processor.discrete_features + processor.categorical_features:
        if feature in actual_values:
            actual = actual_values[feature]
            probs = prediction_probs[feature][0]
            
            if feature in processor.continuous_features:
                # Find which bin contains the actual value
                bin_edges = processor.discretizers[feature].bin_edges_[0]
                bin_idx = np.digitize(actual, bin_edges) - 1
                if bin_idx >= len(probs):
                    bin_idx = len(probs) - 1
                prob = probs[bin_idx]
            else:
                # For discrete features, use the probability directly
                prob = probs[int(actual)]
            
            # Store the probability for this feature
            feature_probabilities[feature] = prob
            
            # Convert probability to anomaly score (0 = normal, 1 = anomalous)
            feature_scores[feature] = 1 - prob
    
    # Calculate total probability of the transaction falling within the pattern
    total_probability = np.prod(list(feature_probabilities.values()))
    
    # Overall anomaly score is the weighted average of feature scores
    weights = {
        'amt': 0.3,  # Amount is very important
        'hour': 0.1,
        'day': 0.05,
        'month': 0.05,
        'dayofweek': 0.1,
        'merchant_encoded': 0.2,  # Merchant is important
        'category_encoded': 0.1,
        'lat': 0.05,
        'long': 0.05,
        'merch_lat': 0.05,
        'merch_long': 0.05
    }
    
    total_score = sum(feature_scores[f] * weights[f] for f in feature_scores)
    total_weight = sum(weights[f] for f in feature_scores)
    
    return total_score / total_weight, feature_scores, feature_probabilities, total_probability

def predict_next_transaction(model, user_transactions, sequence_length, processor, fraud_transactions=[]):
    """Predict the next transaction with probability distributions for each feature."""
    sequence = prepare_sequence(user_transactions, sequence_length, processor)
    
    predictions = model.predict(sequence)
    
    features_to_plot = processor.all_features()
    features_to_plot.remove('day')
    features_to_plot.remove('month')
    features_to_plot.remove('dayofweek')
    
    n_rows = (len(features_to_plot) + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(features_to_plot):
        plot_feature_distribution(processor, feature, predictions, ax=axes[i], 
                                historical_data=user_transactions,
                                fraud_data=fraud_transactions)
    
    # Hide empty subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('prediction_distributions.png', bbox_inches='tight')
    plt.close()
    
    # Get most likely values for each feature
    most_likely_values = {
        feature: get_most_likely_value(processor, feature, predictions)
        for feature in features_to_plot
    }
    
    # Create a summary dictionary
    prediction_summary = {
        'most_likely_values': most_likely_values,
        'raw_distributions': predictions
    }
    
    return prediction_summary

def interpret_prediction(prediction_summary, processor, threshold=0.7):
    """Interpret the prediction results and provide insights."""
    insights = []
    
    # Analyze amount distribution
    amt_probs = prediction_summary['raw_distributions']['amt'][0]
    amt_entropy = -np.sum(amt_probs * np.log2(amt_probs + 1e-10))
    max_entropy = np.log2(len(amt_probs))
    amt_uncertainty = amt_entropy / max_entropy
    
    if amt_uncertainty > 0.8:
        insights.append("⚠️ High uncertainty in predicted amount")
    
    # Analyze time patterns
    hour_probs = prediction_summary['raw_distributions']['hour'][0]
    most_likely_hour = prediction_summary['most_likely_values']['hour']['avg']
    if most_likely_hour < 6 or most_likely_hour > 22:
        insights.append(f"⚠️ Unusual transaction hour: {int(most_likely_hour)}:00")
    
    # Analyze location
    lat = prediction_summary['most_likely_values']['lat']['avg']
    long = prediction_summary['most_likely_values']['long']['avg']
    merch_lat = prediction_summary['most_likely_values']['merch_lat']['avg']
    merch_long = prediction_summary['most_likely_values']['merch_long']['avg']
    
    # Calculate distance between user and merchant
    from math import radians, sin, cos, sqrt, atan2
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance = R * c
        
        return distance
    
    distance = haversine_distance(lat, long, merch_lat, merch_long)
    if distance > 100:  # More than 100km
        insights.append(f"⚠️ Large distance between user and merchant: {distance:.1f}km")
    
    return insights

def predict_and_analyze(model, user_transactions, sequence_length, processor, actual_transaction=None, fraud_transactions=[]):
    """Predict next transaction and provide comprehensive analysis."""
    
    # Get prediction with distributions
    prediction = predict_next_transaction(model, user_transactions, sequence_length, processor, fraud_transactions)
    
    # Get interpretation insights
    insights = interpret_prediction(prediction, processor)
    
    # If we have the actual transaction, calculate anomaly scores
    if actual_transaction is not None:
        anomaly_score, feature_scores, feature_probabilities, total_probability = calculate_anomaly_score(
            prediction['raw_distributions'],
            actual_transaction,
            processor
        )
        
        # Add anomaly insights
        if anomaly_score > 0.7:
            insights.append(f"⚠️ High overall anomaly score: {anomaly_score:.2f}")
            # Add specific feature anomalies
            for feature, score in feature_scores.items():
                if score > 0.8:
                    insights.append(f"⚠️ Unusual {feature}: {score:.2f}")
        
        # Add probability insights
        insights.append(f"📊 Total probability of transaction falling within pattern: {total_probability:.4f}")
        for feature, prob in feature_probabilities.items():
            if prob < 0.1:  # Very low probability features
                insights.append(f"⚠️ Low probability for {feature}: {prob:.4f}")
    
    return {
        'prediction': prediction,
        'insights': insights,
        'anomaly_score': anomaly_score if actual_transaction is not None else None,
        'feature_scores': feature_scores if actual_transaction is not None else None,
        'feature_probabilities': feature_probabilities if actual_transaction is not None else None,
        'total_probability': total_probability if actual_transaction is not None else None
    }

def load_model_and_predict_all(model_path='models/lstm_transaction_model.h5', encoders_path='models/categorical_encoders.joblib'):
    lstm_model, encoders = load_lstm_model(model_path, encoders_path)
    
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
                next_transaction = predict_and_analyze(
                    lstm_model, user_transactions, sequence_length, encoders
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