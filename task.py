import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

def load_and_preprocess_files(file_pattern, group_label):
    # List all files matching the pattern
    files = glob.glob(file_pattern)
    
    # Initialize a dictionary to hold DataFrames
    dataframes = {}
    
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
        
        # Use the file name as the key in the dictionary
        file_name = file.split('/')[-1]  # Extract the file name from the path
        dataframes[file_name] = df_scaled
    
    return dataframes

def get_important_features(df):
    X = df.drop('group', axis=1)
    y = df['group']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train[0:100], y_train[0:100])

    # Get feature importances
    importances = rf_model.feature_importances_
    feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)

    # Print top 10 features
    print("Top 10 features:")
    print(feature_importances.head(10))

    # Select top N features (e.g., top 10)
    N = 10
    top_features = feature_importances.nlargest(N).index.tolist()

    top_features_ints = list(map(int, top_features)).sort()

    return (top_features_ints)

def classify_with_kmeans(X, y):
    # Step 2: K-means Clustering

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Perform K-means clustering
    n_clusters = 2  # Assuming binary classification (ADHD vs Control)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_train_scaled)

    # Predict clusters for training and test data
    train_clusters = kmeans.predict(X_train_scaled)
    test_clusters = kmeans.predict(X_test_scaled)

    # Step 3: Evaluation

    # Function to assign cluster labels to original classes
    def assign_labels(clusters, true_labels):
        label_mapping = {}
        for i in range(n_clusters):
            mask = (clusters == i)
            label_mapping[i] = true_labels[mask].mode()[0]
        return np.array([label_mapping[c] for c in clusters])

    # Assign labels and calculate accuracy
    train_labels = assign_labels(train_clusters, y_train)
    test_labels = assign_labels(test_clusters, y_test)

    train_accuracy = accuracy_score(y_train, train_labels)
    test_accuracy = accuracy_score(y_test, test_labels)

    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, test_labels)
    print("Confusion Matrix:")
    print(cm)

    # Visualize results (for 2D projection)
    if len(top_features) >= 2:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=test_clusters, cmap='viridis')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, linewidths=3, color='r')
        plt.title('K-means Clustering Results')
        plt.colorbar(scatter)
        plt.show()

def reduce_features(dataframes_dict, selected_features):
    for filename, dataframe in dataframes_dict.items():
        dataframes_dict[filename] = dataframe.iloc[:, selected_features]

if __name__ == '__main__':
    # Define the file patterns and group labels
    adhd_pattern = r'C:\Users\97253\Desktop\לימודים\סמסטרים\סמסטר 8\חישה ולמידה\adhdcsv\*'
    control_pattern = r'C:\Users\97253\Desktop\לימודים\סמסטרים\סמסטר 8\חישה ולמידה\controlcsv\*'

    # Load and preprocess files for each group
    adhd_dataframes = load_and_preprocess_files(adhd_pattern, 'ADHD')
    control_dataframes = load_and_preprocess_files(control_pattern, 'Control')

    # Combine all DataFrames into a single DataFrame (if needed)
    combined_df = pd.concat(list(adhd_dataframes.values()) + list(control_dataframes.values()), ignore_index=True)

    top_features = get_important_features(combined_df)
    reduce_features(adhd_dataframes, top_features)
    reduce_features(control_dataframes, top_features)

    print()

    # classify_with_kmeans(X_selected, y_selected)
