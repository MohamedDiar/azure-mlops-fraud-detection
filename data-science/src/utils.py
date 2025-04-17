# data-science/src/utils.py
# Contains shared functions adapted from the user's notebooks

import os
import datetime
import time
import pickle
import json
import pandas as pd
import numpy as np

import sklearn
from sklearn import metrics, preprocessing, model_selection, pipeline, tree, ensemble, linear_model

# === Data Loading/Saving ===

def save_object(obj, filename):
    """Saves object as pickle file."""
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# === Data Preprocessing & Transformation ===

def is_weekend(tx_datetime):
    """Checks if a datetime object falls on a weekend."""
    weekday = tx_datetime.weekday() # Monday is 0 and Sunday is 6
    return int(weekday >= 5)

def is_night(tx_datetime):
    """Checks if a datetime object's time is between 00:00 and 06:00."""
    tx_hour = tx_datetime.hour
    return int(tx_hour <= 6)

def get_customer_spending_behaviour_features(customer_transactions, windows_size_in_days=[1,7,30]):
    """Calculates customer spending behavior features over specified windows."""
    # Let us first order transactions chronologically
    customer_transactions=customer_transactions.sort_values('TX_DATETIME')

    # The transaction date and time is set as the index, which will allow the use of the rolling function
    # Ensure index is unique if TX_DATETIME has duplicates, append a counter or use Transaction ID if unique per customer
    customer_transactions.index=customer_transactions.TX_DATETIME

    # For each window size
    for window_size in windows_size_in_days:

        # Compute the sum of the transaction amounts and the number of transactions for the given window size
        # Rolling sums/counts include the current transaction
        SUM_AMOUNT_TX_WINDOW=customer_transactions['TX_AMOUNT'].rolling(str(window_size)+'d').sum()
        NB_TX_WINDOW=customer_transactions['TX_AMOUNT'].rolling(str(window_size)+'d').count()

        # Compute the average transaction amount for the given window size
        # NB_TX_WINDOW is always >=1 since current transaction is always included
        AVG_AMOUNT_TX_WINDOW=SUM_AMOUNT_TX_WINDOW/NB_TX_WINDOW

        # Save feature values
        customer_transactions['CUSTOMER_ID_NB_TX_'+str(window_size)+'DAY_WINDOW']=list(NB_TX_WINDOW)
        customer_transactions['CUSTOMER_ID_AVG_AMOUNT_'+str(window_size)+'DAY_WINDOW']=list(AVG_AMOUNT_TX_WINDOW)

    # Reindex according to transaction IDs (assuming TRANSACTION_ID is unique)
    customer_transactions.index=customer_transactions.TRANSACTION_ID

    # And return the dataframe with the new features
    return customer_transactions

def get_count_risk_rolling_window(terminal_transactions, delay_period=7, windows_size_in_days=[1,7,30], feature="TERMINAL_ID"):
    """Calculates terminal risk features over specified windows with a delay."""
    terminal_transactions=terminal_transactions.sort_values('TX_DATETIME')

    # Set index for rolling calculation
    terminal_transactions.index=terminal_transactions.TX_DATETIME

    # Calculate frauds and transactions within the delay period immediately preceding the current transaction
    NB_FRAUD_DELAY=terminal_transactions['TX_FRAUD'].rolling(str(delay_period)+'d').sum()
    NB_TX_DELAY=terminal_transactions['TX_FRAUD'].rolling(str(delay_period)+'d').count()

    # Calculate frauds and transactions for the window size plus the delay period
    for window_size in windows_size_in_days:

        NB_FRAUD_DELAY_WINDOW=terminal_transactions['TX_FRAUD'].rolling(str(delay_period+window_size)+'d').sum()
        NB_TX_DELAY_WINDOW=terminal_transactions['TX_FRAUD'].rolling(str(delay_period+window_size)+'d').count()

        # Calculate frauds and transactions for the specific window (shifted back by delay)
        NB_FRAUD_WINDOW=NB_FRAUD_DELAY_WINDOW-NB_FRAUD_DELAY
        NB_TX_WINDOW=NB_TX_DELAY_WINDOW-NB_TX_DELAY

        # Calculate risk score for the window
        # Avoid division by zero: if NB_TX_WINDOW is 0, risk is 0
        RISK_WINDOW = (NB_FRAUD_WINDOW / NB_TX_WINDOW).fillna(0)

        terminal_transactions[feature+'_NB_TX_'+str(window_size)+'DAY_WINDOW']=list(NB_TX_WINDOW)
        terminal_transactions[feature+'_RISK_'+str(window_size)+'DAY_WINDOW']=list(RISK_WINDOW)

    # Restore original index
    terminal_transactions.index=terminal_transactions.TRANSACTION_ID

    # Replace any remaining NA values (should not happen with fillna(0) above, but as safeguard)
    terminal_transactions.fillna(0,inplace=True)

    return terminal_transactions

# === Train/Test Splitting ===

def get_train_test_set(transactions_df,
                       start_date_training,
                       delta_train=7,delta_delay=7,delta_test=7,
                       sampling_ratio=1.0, # Not used in final model pipeline, kept for function consistency
                       random_state=0):    # Not used in final model pipeline, kept for function consistency
    """Gets Train and Test sets for a given start date and time deltas."""
    # Validate inputs
    required_cols = ['TX_DATETIME', 'TX_TIME_DAYS', 'CUSTOMER_ID', 'TX_FRAUD', 'TRANSACTION_ID']
    if not all(col in transactions_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in transactions_df.columns]
        raise ValueError(f"Missing required columns in transactions_df for get_train_test_set: {missing}")
    if not isinstance(start_date_training, datetime.datetime):
         raise ValueError("start_date_training must be a datetime object")

    # Get the training set data based on TX_DATETIME
    train_df = transactions_df[
        (transactions_df.TX_DATETIME >= start_date_training) &
        (transactions_df.TX_DATETIME < start_date_training + datetime.timedelta(days=delta_train))
    ]

    # Get the test set data, applying the delay logic
    test_df_list = []

    # First, get known defrauded customers from the training set
    known_defrauded_customers = set(train_df[train_df.TX_FRAUD == 1].CUSTOMER_ID)

    # Get the relative starting day of training set (using TX_TIME_DAYS)
    if train_df.empty:
        print(f"Warning (get_train_test_set): Training period starting {start_date_training.strftime('%Y-%m-%d')} is empty.")
        # Return empty DataFrames matching expected columns
        return (train_df.copy(), pd.DataFrame(columns=transactions_df.columns))

    start_tx_time_days_training = train_df.TX_TIME_DAYS.min()

    # Iterate through each day of the test period delta
    for day in range(delta_test):
        # Target day for test data
        current_test_day = start_tx_time_days_training + delta_train + delta_delay + day
        test_df_day = transactions_df[transactions_df.TX_TIME_DAYS == current_test_day]

        # Day to check for newly compromised cards (test_day - delay_period)
        # The original logic seems to use 'day-1' relative to the end of the training period start
        delay_check_day = start_tx_time_days_training + delta_train + day - 1 # Check this logic carefully matches original intent
        compromised_cards_delay_period = transactions_df[transactions_df.TX_TIME_DAYS == delay_check_day]

        new_defrauded_customers = set(compromised_cards_delay_period[compromised_cards_delay_period.TX_FRAUD == 1].CUSTOMER_ID)
        known_defrauded_customers.update(new_defrauded_customers) # Use update for sets

        # Filter the current day's test data
        test_df_day_filtered = test_df_day[~test_df_day.CUSTOMER_ID.isin(known_defrauded_customers)]
        test_df_list.append(test_df_day_filtered)

    # Concatenate daily test sets
    if not test_df_list:
        print(f"Warning (get_train_test_set): Test period resulted in an empty set after filtering for training start {start_date_training.strftime('%Y-%m-%d')}.")
        test_df_final = pd.DataFrame(columns=transactions_df.columns)
    else:
        test_df_final = pd.concat(test_df_list, ignore_index=True)


    # Sampling - Not typically used in the final pipeline run, but part of the original function
    if sampling_ratio < 1.0:
        train_df_frauds = train_df[train_df.TX_FRAUD==1].sample(frac=sampling_ratio, random_state=random_state)
        train_df_genuine = train_df[train_df.TX_FRAUD==0].sample(frac=sampling_ratio, random_state=random_state)
        train_df = pd.concat([train_df_frauds, train_df_genuine])

    # Sort final data sets by ascending order of transaction ID
    train_df = train_df.sort_values('TRANSACTION_ID').reset_index(drop=True)
    test_df_final = test_df_final.sort_values('TRANSACTION_ID').reset_index(drop=True)

    return (train_df, test_df_final)


def prequentialSplit_with_dates(transactions_df,
                                start_date_training,
                                n_folds=4,
                                delta_train=7,
                                delta_delay=7,
                                delta_assessment=7):
    """Generates prequential splits, returning indices and printing date ranges for each fold."""
    prequential_split_indices = []
    print(f"\n--- Generating Prequential Folds (n_folds={n_folds}) ---")
    print(f"Base Start Date (Fold 0 Train Start): {start_date_training.strftime('%Y-%m-%d')}")
    print(f"Deltas: Train={delta_train}, Delay={delta_delay}, Assessment={delta_assessment}")
    print("-" * 60)

    for fold in range(n_folds):
        start_date_training_fold = start_date_training - datetime.timedelta(days=fold * delta_assessment)
        end_date_training_fold = start_date_training_fold + datetime.timedelta(days=delta_train)
        start_date_test_fold = end_date_training_fold + datetime.timedelta(days=delta_delay)
        end_date_test_fold = start_date_test_fold + datetime.timedelta(days=delta_assessment)

        inclusive_end_train = end_date_training_fold - datetime.timedelta(days=1)
        inclusive_end_test = end_date_test_fold - datetime.timedelta(days=1)

        print(f"Fold {fold}:")
        print(f"  Train Period: {start_date_training_fold.strftime('%Y-%m-%d')} to {inclusive_end_train.strftime('%Y-%m-%d')}")
        print(f"  Test Period:  {start_date_test_fold.strftime('%Y-%m-%d')} to {inclusive_end_test.strftime('%Y-%m-%d')}")

        try:
            (train_df, test_df) = get_train_test_set(transactions_df,
                                                   start_date_training=start_date_training_fold,
                                                   delta_train=delta_train,
                                                   delta_delay=delta_delay,
                                                   delta_test=delta_assessment)
        except Exception as e:
            print(f"  -> ERROR calling get_train_test_set for fold {fold}: {e}")
            print(f"     Skipping fold {fold}.")
            print("-" * 10)
            continue

        if not train_df.empty and not test_df.empty:
            # Use intersection to ensure indices are valid in the original df passed to GridSearchCV
            valid_train_indices = transactions_df.index.intersection(train_df.index)
            valid_test_indices = transactions_df.index.intersection(test_df.index)
            # Convert to lists for GridSearchCV compatibility
            indices_train = list(valid_train_indices)
            indices_test = list(valid_test_indices)

            if indices_train and indices_test: # Ensure both lists are non-empty after intersection
                 prequential_split_indices.append((indices_train, indices_test))
                 print(f"  -> Train size: {len(indices_train)}, Test size: {len(indices_test)}. Added fold indices.")
            else:
                 print(f"  -> Warning: Fold {fold} resulted in empty train or test indices after intersection. Skipping.")
        else:
             print(f"  -> Warning: Fold {fold} generated empty train ({train_df.shape}) or test ({test_df.shape}) set. Skipping fold.")
        print("-" * 10)

    if not prequential_split_indices:
        print(f"Warning (prequentialSplit): No valid folds generated for start date {start_date_training.strftime('%Y-%m-%d')} and {n_folds} folds.")

    print("--- Finished Generating Prequential Folds ---")
    return prequential_split_indices


# === Performance Assessment ===

def card_precision_top_k_day(df_day,top_k):
    """Computes card precision top k for a single day."""
    # Ensure required columns are present
    required = ['CUSTOMER_ID', 'predictions', 'TX_FRAUD']
    if not all(col in df_day.columns for col in required):
        missing = [col for col in required if col not in df_day.columns]
        print(f"Warning (card_precision_top_k_day): Missing columns {missing}. Returning empty list, 0.")
        return [], 0.0
    if df_day.empty:
        return [], 0.0

    # Group by customer, take max prediction and fraud flag
    df_day_grouped = df_day.groupby('CUSTOMER_ID').agg(
        {'predictions': 'max', 'TX_FRAUD': 'max'}
    ).sort_values(by="predictions", ascending=False).reset_index()

    # Get top k customers
    df_day_top_k = df_day_grouped.head(top_k)
    # Get list of customer IDs from the top k that are actually fraudulent
    list_detected_compromised_cards = list(df_day_top_k[df_day_top_k.TX_FRAUD == 1].CUSTOMER_ID)

    # Compute precision, handle k=0
    if top_k > 0:
        card_precision_top_k = len(list_detected_compromised_cards) / top_k
    else:
        card_precision_top_k = 0.0

    return list_detected_compromised_cards, card_precision_top_k


def card_precision_top_k(predictions_df, top_k, remove_detected_compromised_cards=True):
    """Computes average card precision top k over multiple days."""
    required = ['TX_TIME_DAYS', 'CUSTOMER_ID', 'predictions', 'TX_FRAUD']
    if not all(col in predictions_df.columns for col in required):
        missing = [col for col in required if col not in predictions_df.columns]
        raise ValueError(f"Missing required columns in predictions_df for card_precision_top_k: {missing}")

    list_days=sorted(predictions_df['TX_TIME_DAYS'].unique())
    list_detected_compromised_cards = []
    card_precision_top_k_per_day_list = []
    nb_compromised_cards_per_day = [] # For reference

    for day in list_days:
        df_day = predictions_df[predictions_df['TX_TIME_DAYS'] == day].copy()
        # Filter out already detected cards if required
        if remove_detected_compromised_cards:
             df_day = df_day[~df_day.CUSTOMER_ID.isin(list_detected_compromised_cards)]

        if df_day.empty:
             nb_compromised_cards_per_day.append(0)
             card_precision_top_k_per_day_list.append(0.0)
             continue

        # Keep track of total compromised cards on this day (before filtering for top k)
        nb_compromised_cards_per_day.append(len(df_day[df_day.TX_FRAUD == 1].CUSTOMER_ID.unique()))

        # Calculate daily precision
        detected_compromised_cards, card_precision_top_k_daily = card_precision_top_k_day(
            df_day[['CUSTOMER_ID', 'predictions', 'TX_FRAUD']], top_k
        )
        card_precision_top_k_per_day_list.append(card_precision_top_k_daily)

        # Update list of detected cards for next day's filtering
        if remove_detected_compromised_cards:
            list_detected_compromised_cards.extend(detected_compromised_cards)
            list_detected_compromised_cards = list(set(list_detected_compromised_cards)) # Keep unique

    # Compute the mean, handle empty list
    mean_card_precision_top_k = np.mean(card_precision_top_k_per_day_list) if card_precision_top_k_per_day_list else 0.0

    return nb_compromised_cards_per_day, card_precision_top_k_per_day_list, mean_card_precision_top_k


def card_precision_top_k_custom(y_true, y_pred, top_k, transactions_df):
    """Scorer function for GridSearchCV using Card Precision Top K."""
    # Ensure inputs are usable
    if not isinstance(y_true, pd.Series) or not isinstance(transactions_df, pd.DataFrame):
         print("Warning (CP@k scorer): y_true must be Series, transactions_df must be DataFrame.")
         return 0.0 # Return default score on error
    if len(y_pred) != len(y_true):
        print("Warning (CP@k scorer): y_pred and y_true lengths differ.")
        return 0.0
    if transactions_df.empty:
         print("Warning (CP@k scorer): Scorer helper transactions_df is empty.")
         return 0.0

    # Get indices from y_true (representing the current fold)
    current_fold_indices = y_true.index

    # Select the relevant transactions from the helper dataframe
    # Use intersection to handle cases where index might not exist in helper (shouldn't happen if prepared correctly)
    valid_indices = current_fold_indices.intersection(transactions_df.index)
    if valid_indices.empty:
        print(f"Warning (CP@k scorer): No matching indices found in scorer helper DF for the current fold ({len(current_fold_indices)} indices).")
        return 0.0

    predictions_df_fold = transactions_df.loc[valid_indices].copy()

    # Add predictions, ensuring alignment using the original fold indices
    y_pred_series = pd.Series(y_pred, index=current_fold_indices)
    # Select only the predictions corresponding to the valid indices found in the scorer helper
    predictions_df_fold['predictions'] = y_pred_series.loc[valid_indices]

    # Compute the CP@k metric using the main function
    try:
        _, _, mean_card_precision_top_k = card_precision_top_k(predictions_df_fold, top_k)
    except Exception as e:
        print(f"Error calculating CP@{top_k} within scorer: {e}")
        return 0.0 # Return default score on calculation error

    return mean_card_precision_top_k


def performance_assessment(predictions_df, output_feature='TX_FRAUD',
                           prediction_feature='predictions', top_k_list=[100],
                           rounded=True):
    """Calculates standard and custom performance metrics."""
    required = [output_feature, prediction_feature]
    if not all(col in predictions_df.columns for col in required):
        missing = [col for col in required if col not in predictions_df.columns]
        raise ValueError(f"Missing required columns for performance_assessment: {missing}")

    y_true = predictions_df[output_feature]
    y_pred_proba = predictions_df[prediction_feature]

    AUC_ROC = np.nan
    AP = np.nan
    if len(y_true.unique()) > 1: # Check for multiple classes
        try:
            AUC_ROC = metrics.roc_auc_score(y_true, y_pred_proba)
            AP = metrics.average_precision_score(y_true, y_pred_proba)
        except ValueError as e:
            print(f"Warning (performance_assessment): ValueError calculating AUC/AP: {e}")
    else:
        print("Warning (performance_assessment): Only one class present. AUC ROC and Average Precision are undefined.")

    performances = pd.DataFrame([[AUC_ROC, AP]], columns=['AUC ROC', 'Average precision'])

    # Add CP@k metric(s)
    cpk_required = ['TX_TIME_DAYS', 'CUSTOMER_ID']
    if all(col in predictions_df.columns for col in cpk_required):
        for top_k in top_k_list:
            try:
                _, _, mean_card_precision_top_k = card_precision_top_k(predictions_df, top_k)
                performances[f'Card Precision@{top_k}'] = mean_card_precision_top_k
            except Exception as e:
                 print(f"Warning (performance_assessment): Error calculating CP@{top_k}: {e}")
                 performances[f'Card Precision@{top_k}'] = np.nan
    else:
        missing_cpk = [col for col in cpk_required if col not in predictions_df.columns]
        print(f"Warning (performance_assessment): Skipping Card Precision@k calculation due to missing columns: {missing_cpk}.")
        for top_k in top_k_list:
             performances[f'Card Precision@{top_k}'] = np.nan

    if rounded:
        performances = performances.round(3)

    return performances


def get_class_from_fraud_probability(fraud_probabilities, threshold=0.5):
    """Converts probabilities to binary classes based on a threshold."""
    predicted_classes = [1 if p >= threshold else 0 for p in fraud_probabilities]
    return predicted_classes


def threshold_based_metrics(fraud_probabilities, true_label, thresholds_list):
    """Calculates various threshold-based metrics."""
    results = []
    for threshold in thresholds_list:
        predicted_classes = get_class_from_fraud_probability(fraud_probabilities, threshold=threshold)
        try:
            cm = metrics.confusion_matrix(true_label, predicted_classes, labels=[0, 1])
            TN, FP, FN, TP = cm.ravel()
        except ValueError as e:
            print(f"Warning (threshold_metrics): Confusion matrix error for threshold {threshold}: {e}")
            TN, FP, FN, TP = 0, 0, 0, 0

        # Calculations with division-by-zero handling
        total = TN + FP + FN + TP
        MME = (FP + FN) / total if total > 0 else 0
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
        FPR = FP / (TN + FP) if (TN + FP) > 0 else 0
        FNR = FN / (TP + FN) if (TP + FN) > 0 else 0
        BER = 0.5 * (FPR + FNR)
        Gmean = np.sqrt(TPR * TNR) if TPR >= 0 and TNR >= 0 else 0 # Ensure non-negative for sqrt
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        NPV = TN / (TN + FN) if (TN + FN) > 0 else 0
        FDR = FP / (TP + FP) if (TP + FP) > 0 else 0
        FOR = FN / (TN + FN) if (TN + FN) > 0 else 0
        F1_score = 2 * (precision * TPR) / (precision + TPR) if (precision + TPR) > 0 else 0

        results.append([threshold, MME, TPR, TNR, FPR, FNR, BER, Gmean, precision, NPV, FDR, FOR, F1_score])

    results_df = pd.DataFrame(results, columns=['Threshold', 'MME', 'TPR', 'TNR', 'FPR', 'FNR', 'BER', 'G-mean', 'Precision', 'NPV', 'FDR', 'FOR', 'F1 Score'])
    return results_df


# === Model Selection ===

def params_to_str(params):
    """Helper function to convert parameter dict to a readable, sorted string summary."""
    if not isinstance(params, dict):
        return str(params)
    try:
         items = [f"{k.split('__')[1]}={v}" for k, v in sorted(params.items())]
         return ", ".join(items)
    except Exception:
         return str(params) # Fallback


def prequential_grid_search(transactions_df,
                            classifier,
                            input_features, output_feature,
                            parameters, scoring,
                            start_date_training,
                            n_folds=4,
                            expe_type='Test',
                            delta_train=7,
                            delta_delay=7,
                            delta_assessment=7,
                            performance_metrics_list_grid=['roc_auc'],
                            performance_metrics_list=['AUC ROC'],
                            n_jobs=-1):
    """Performs GridSearchCV using prequential splitting."""
    # Input validation
    if transactions_df.empty:
         print(f"ERROR (prequential_grid_search): Input transactions_df is empty for {expe_type}.")
         return pd.DataFrame() # Return empty matching expected structure later
    if not scoring:
         print(f"ERROR (prequential_grid_search): scoring dictionary is empty for {expe_type}.")
         return pd.DataFrame()

    # Create pipeline
    estimators = [('scaler', sklearn.preprocessing.StandardScaler()), ('clf', classifier)]
    pipe = sklearn.pipeline.Pipeline(estimators)

    # Generate prequential splits (returns list of (train_indices, test_indices))
    prequential_split_indices = prequentialSplit_with_dates(transactions_df,
                                                        start_date_training=start_date_training,
                                                        n_folds=n_folds,
                                                        delta_train=delta_train,
                                                        delta_delay=delta_delay,
                                                        delta_assessment=delta_assessment)

    if not prequential_split_indices:
         print(f"ERROR (prequential_grid_search): No valid prequential splits generated for {expe_type}. Cannot run GridSearchCV.")
         return pd.DataFrame() # Return empty matching expected structure later

    # Setup GridSearchCV
    grid_search = sklearn.model_selection.GridSearchCV(pipe, parameters, scoring=scoring,
                                                       cv=prequential_split_indices, # Use generated indices
                                                       refit=False, # We only care about CV results here
                                                       n_jobs=n_jobs,
                                                       return_train_score=False)

    X = transactions_df[input_features]
    y = transactions_df[output_feature]

    # Handle NaNs (Pipeline's scaler will handle or raise error)
    if X.isnull().values.any():
        print(f"Warning (prequential_grid_search): NaNs detected in features for {expe_type}. Pipeline's StandardScaler should handle this.")
        # If imputation outside pipeline is needed, add it here.

    print(f"Starting GridSearchCV for {expe_type} set (Classifier: {classifier.__class__.__name__})...")
    try:
        grid_search.fit(X, y)
    except Exception as e:
        print(f"ERROR (prequential_grid_search): GridSearchCV fit failed for {expe_type}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame() # Return empty on failure

    print(f"Finished GridSearchCV for {expe_type} set.")

    # Extract results
    performances_df = pd.DataFrame()
    cv_results = grid_search.cv_results_

    # Map grid keys to display names (assuming lists align)
    metric_mapping = dict(zip(performance_metrics_list_grid, performance_metrics_list))

    for grid_key, display_name in metric_mapping.items():
        mean_score_key = f'mean_test_{grid_key}'
        std_score_key = f'std_test_{grid_key}'

        if mean_score_key in cv_results:
            performances_df[f'{display_name} {expe_type}'] = cv_results[mean_score_key]
        else:
            print(f"Warning: Mean score key '{mean_score_key}' not found in cv_results_ for {expe_type}.")
            performances_df[f'{display_name} {expe_type}'] = np.nan

        if std_score_key in cv_results:
            performances_df[f'{display_name} {expe_type} Std'] = cv_results[std_score_key]
        else:
            # print(f"Note: Std score key '{std_score_key}' not found in cv_results_ for {expe_type}.") # Less verbose
            performances_df[f'{display_name} {expe_type} Std'] = np.nan # Use NaN for missing std


    if 'params' in cv_results:
        performances_df['Parameters'] = cv_results['params']
        performances_df['Parameters summary'] = performances_df['Parameters'].apply(params_to_str)
    else:
        print("Warning: 'params' key not found in cv_results_. Adding empty parameter columns.")
        num_rows = len(next(iter(cv_results.values()))) # Get length from another column
        performances_df['Parameters'] = [{} for _ in range(num_rows)]
        performances_df['Parameters summary'] = 'N/A'


    if 'mean_fit_time' in cv_results:
        performances_df['Execution time'] = cv_results['mean_fit_time']
    else:
        print("Warning: 'mean_fit_time' key not found in cv_results_.")
        performances_df['Execution time'] = np.nan

    return performances_df


def model_selection_wrapper(transactions_df,
                            classifier,
                            input_features, output_feature,
                            parameters,
                            scoring,
                            start_date_training_for_valid,
                            start_date_training_for_test,
                            n_folds=4,
                            delta_train=7,
                            delta_delay=7,
                            delta_assessment=7,
                            performance_metrics_list_grid=['roc_auc'],
                            performance_metrics_list=['AUC ROC'],
                            n_jobs=-1):
    """Wraps prequential grid search for validation and test estimation."""
    # Get performances on the validation set
    print("--- Running Prequential Grid Search for Validation Set ---")
    performances_df_validation = prequential_grid_search(
        transactions_df, classifier,
        input_features, output_feature,
        parameters, scoring,
        start_date_training=start_date_training_for_valid,
        n_folds=n_folds, expe_type='Validation',
        delta_train=delta_train, delta_delay=delta_delay, delta_assessment=delta_assessment,
        performance_metrics_list_grid=performance_metrics_list_grid,
        performance_metrics_list=performance_metrics_list, n_jobs=n_jobs
    )

    # Get performances on the test set estimation
    print("--- Running Prequential Grid Search for Test Set Estimation ---")
    performances_df_test = prequential_grid_search(
        transactions_df, classifier,
        input_features, output_feature,
        parameters, scoring,
        start_date_training=start_date_training_for_test,
        n_folds=n_folds, expe_type='Test',
        delta_train=delta_train, delta_delay=delta_delay, delta_assessment=delta_assessment,
        performance_metrics_list_grid=performance_metrics_list_grid,
        performance_metrics_list=performance_metrics_list, n_jobs=n_jobs
    )

    # Merge results
    if performances_df_test.empty and performances_df_validation.empty:
        print("Warning (model_selection_wrapper): Both Test and Validation results are empty.")
        return pd.DataFrame()
    elif performances_df_test.empty:
         print("Warning (model_selection_wrapper): Test results are empty. Returning only Validation.")
         return performances_df_validation
    elif performances_df_validation.empty:
         print("Warning (model_selection_wrapper): Validation results are empty. Returning only Test.")
         return performances_df_test
    else:
        # Merge, ensuring 'Parameters summary' exists
        if 'Parameters summary' not in performances_df_test.columns or 'Parameters summary' not in performances_df_validation.columns:
             print("ERROR (model_selection_wrapper): 'Parameters summary' missing. Cannot merge. Returning Test results.")
             return performances_df_test

        # Drop redundant columns from validation before merge
        val_cols_to_drop = ['Parameters', 'Execution time']
        validation_subset = performances_df_validation.drop(columns=val_cols_to_drop, errors='ignore')
        # Outer merge preserves all parameter sets tested
        performances_df_merged = pd.merge(performances_df_test, validation_subset, on='Parameters summary', how='outer')
        return performances_df_merged


def get_summary_performances(performances_df, parameter_column_name="Parameters summary"):
    """Summarizes performance dataframe to find best parameters based on validation metrics."""
    metrics_list = ['AUC ROC', 'Average precision', 'Card Precision@100'] # Assuming CP@100 was the target
    performances_results = pd.DataFrame(columns=metrics_list)

    if performances_df.empty:
        print("Warning: Empty performance dataframe passed to get_summary_performances.")
        # Return structure with N/A
        na_vals = ['N/A'] * len(metrics_list)
        for row_name in ["Best estimated parameters", "Validation performance", "Test performance", "Optimal parameter(s)", "Optimal test performance"]:
             performances_results.loc[row_name] = na_vals
        return performances_results

    # Ensure index is reset for iloc usage
    performances_df = performances_df.reset_index(drop=True)

    best_estimated_parameters = []
    validation_performance = []
    test_performance = []

    for metric in metrics_list:
        val_metric_col = metric + ' Validation'
        val_std_col = val_metric_col + ' Std'
        test_metric_col = metric + ' Test'
        test_std_col = test_metric_col + ' Std'

        # Check if required columns exist
        if val_metric_col not in performances_df.columns or test_metric_col not in performances_df.columns:
             print(f"Warning: Missing columns for metric {metric}. Adding N/A.")
             best_estimated_parameters.append('N/A')
             validation_performance.append('N/A')
             test_performance.append('N/A')
             continue

        # Find best performance based on validation score (handle NaNs)
        valid_scores = pd.to_numeric(performances_df[val_metric_col], errors='coerce')
        if valid_scores.isna().all():
            print(f"Warning: All validation scores for {metric} are NaN.")
            index_best_validation_performance = 0 # Default to first row
            best_param_summary = performances_df.loc[index_best_validation_performance, parameter_column_name] if parameter_column_name in performances_df.columns else 'N/A'
            val_perf_str = 'NaN'
            # Get test perf at this index
            test_perf_val = pd.to_numeric(performances_df[test_metric_col], errors='coerce').iloc[index_best_validation_performance]
            test_std_val = pd.to_numeric(performances_df.get(test_std_col, np.nan), errors='coerce').iloc[index_best_validation_performance]
            test_perf_str = f"{test_perf_val:.3f} +/- {test_std_val:.2f}" if not pd.isna(test_perf_val) else 'NaN'
        else:
            index_best_validation_performance = valid_scores.idxmax()
            best_param_summary = performances_df.loc[index_best_validation_performance, parameter_column_name] if parameter_column_name in performances_df.columns else 'N/A'
            # Get validation performance string
            val_perf = valid_scores.iloc[index_best_validation_performance]
            val_std = pd.to_numeric(performances_df.get(val_std_col, np.nan), errors='coerce').iloc[index_best_validation_performance]
            val_perf_str = f"{val_perf:.3f} +/- {val_std:.2f}" if not pd.isna(val_perf) else 'NaN'
            # Get test performance string at the same index
            test_perf = pd.to_numeric(performances_df[test_metric_col], errors='coerce').iloc[index_best_validation_performance]
            test_std = pd.to_numeric(performances_df.get(test_std_col, np.nan), errors='coerce').iloc[index_best_validation_performance]
            test_perf_str = f"{test_perf:.3f} +/- {test_std:.2f}" if not pd.isna(test_perf) else 'NaN'

        best_estimated_parameters.append(best_param_summary)
        validation_performance.append(val_perf_str)
        test_performance.append(test_perf_str)

    performances_results.loc["Best estimated parameters"] = best_estimated_parameters
    performances_results.loc["Validation performance"] = validation_performance
    performances_results.loc["Test performance"] = test_performance

    # Find Optimal on Test Set (for reference)
    optimal_parameters = []
    optimal_test_performance = []
    for metric_base in metrics_list:
        test_metric_col = metric_base + ' Test'
        test_std_col = test_metric_col + ' Std'

        if test_metric_col not in performances_df.columns:
            optimal_parameters.append('N/A')
            optimal_test_performance.append('N/A')
            continue

        test_scores = pd.to_numeric(performances_df[test_metric_col], errors='coerce')
        if test_scores.isna().all():
            print(f"Warning: All test scores for {metric_base} are NaN.")
            index_optimal_test_performance = 0 # Default index
            opt_param_summary = performances_df.loc[index_optimal_test_performance, parameter_column_name] if parameter_column_name in performances_df.columns else 'N/A'
            opt_test_perf_str = 'NaN'
        else:
            index_optimal_test_performance = test_scores.idxmax()
            opt_param_summary = performances_df.loc[index_optimal_test_performance, parameter_column_name] if parameter_column_name in performances_df.columns else 'N/A'
            opt_test_perf = test_scores.iloc[index_optimal_test_performance]
            opt_test_std = pd.to_numeric(performances_df.get(test_std_col, np.nan), errors='coerce').iloc[index_optimal_test_performance]
            opt_test_perf_str = f"{opt_test_perf:.3f} +/- {opt_test_std:.2f}" if not pd.isna(opt_test_perf) else 'NaN'

        optimal_parameters.append(opt_param_summary)
        optimal_test_performance.append(opt_test_perf_str)

    performances_results.loc["Optimal parameter(s)"] = optimal_parameters
    performances_results.loc["Optimal test performance"] = optimal_test_performance

    return performances_results