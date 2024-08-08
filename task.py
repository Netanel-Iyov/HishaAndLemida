import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
import math

def flatten_dataframe(df):
    # Flatten the DataFrame and ignore label column
    flattened = df.iloc[:, :-1].values.flatten()

    #extract group label
    group = df['group'][0]
    
    # prepare new dataframe
    flattened_df = pd.DataFrame([flattened])
    flattened_df['group'] = group

    return flattened_df

def load_and_preprocess_files(data):
    # Initialize a list to hold DataFrames
    dataframes = []

    # List all files matching the pattern
    for file_pattern, group_label in data.items():
        files = glob.glob(file_pattern)
        
        # Iterate over each file
        for file in files:
            # Load the CSV file
            df = pd.read_csv(file)
            
            # Handle missing values by filling them with the mean of each column
            df.fillna(df.mean(), inplace=True)
            
            # Ensure all data is in numeric format
            df = df.apply(pd.to_numeric)
            
            # Normalize the data
            scaler = StandardScaler()
            df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
            
            # Add a label column to indicate the group
            df_scaled['group'] = group_label
            
            dataframes.append(df_scaled)
    
    # Determine the minimum number of rows across all dataframes
    # min_rows = min(df.shape[0] for df in dataframes)
    
    minimal_size = 10000

    # Trim each dataframe to the minimum number of rows
    dataframes_trimmed = [df.iloc[:minimal_size] for df in dataframes if df.shape[0] >= minimal_size]
    
    return dataframes_trimmed


def get_important_features(df, n_features):
    X = df.drop('group', axis=1)
    y = df['group']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    n_samples = math.ceil(X_train.shape[0] * 0.2)
    rf_model.fit(X_train[:n_samples], y_train[:n_samples])

    # Get feature importances
    importances = rf_model.feature_importances_
    feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)

    top_features = feature_importances.nlargest(n_features).index.tolist()

    top_features_ints = list(map(int, top_features))
    top_features_ints.sort()

    return top_features_ints

def classify_with_kmeans(X, y):
    # Step 1: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 2: Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 3: Perform K-means clustering
    n_clusters = 2  # Assuming binary classification (ADHD vs Control)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_train_scaled)

    # Step 4: Predict clusters for training and test data
    train_clusters = kmeans.predict(X_train_scaled)
    test_clusters = kmeans.predict(X_test_scaled)

    # Step 5: Map clusters to original labels
    # This mapping is heuristic and assumes the majority label in a cluster is the true label
    def map_clusters_to_labels(clusters, true_labels):
        cluster_to_label = {}
        for cluster in np.unique(clusters):
            mask = (clusters == cluster)
            most_common = true_labels[mask].value_counts().idxmax()
            cluster_to_label[cluster] = most_common
        return np.vectorize(cluster_to_label.get)(clusters)

    train_labels = map_clusters_to_labels(train_clusters, y_train)
    test_labels = map_clusters_to_labels(test_clusters, y_test)

    # Step 6: Calculate accuracy
    train_accuracy = accuracy_score(y_train, train_labels)
    test_accuracy = accuracy_score(y_test, test_labels)

    print(f'train_accuracy, test_accuracy = {train_accuracy}, {test_accuracy}')
    return train_accuracy, test_accuracy


def reduce_features(dataframes, selected_features):
    for i, dataframe in enumerate(dataframes):
        dataframes[i] = dataframe.iloc[:, selected_features]

if __name__ == '__main__':
    # Define the file patterns and group labels
    adhd_pattern = r'C:\Users\97253\Desktop\לימודים\סמסטרים\סמסטר 8\חישה ולמידה\adhdcsv\*'
    control_pattern = r'C:\Users\97253\Desktop\לימודים\סמסטרים\סמסטר 8\חישה ולמידה\controlcsv\*'

    # Load and preprocess files for each group
    dataframes = load_and_preprocess_files({adhd_pattern: 'ADHD', control_pattern: 'Control'})
    
    # Combine all DataFrames into a single DataFrame (if needed)
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Get top features and reduce by it
    n_features = 6
    top_features = get_important_features(combined_df, n_features)
    top_features.append(19)
    reduce_features(dataframes, top_features)

    flattened_dataframes = [flatten_dataframe(df) for df in dataframes] 

    # Combine the flattened dataframes into a single dataframe
    combined_flattened = pd.concat(flattened_dataframes, ignore_index=True)

    X = combined_flattened.drop('group', axis=1)
    y = combined_flattened['group']

    classify_with_kmeans(X, y)
