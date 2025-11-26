import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler


# TO READ PROPERLY HAVE: -FOLDER BEFORE r"...\" OPEN IN VSCODE -HAVE THE SAME FOLDER/FILE NAMES IF YOU RUN ON DIFFERENT DEVICE
try:
    dataset = pd.read_csv("student_performance.csv")
except FileNotFoundError:
    # Fallback
    try:
        dataset = pd.read_csv(r"StudentPerformance_Assignment\student_performance.csv")
    except FileNotFoundError:
        print("Error: File not found. Please make sure 'student_performance.csv' is in the same folder.")
        exit()
dataset = dataset.sample(n=50000, random_state=42)
# print(dataset.describe())  # helps to see some data from the csv
features = ["weekly_self_study_hours", "attendance_percentage", "class_participation", "total_score"]
X = dataset[features]  ## droping 'grade' because its not numerical
scaler = StandardScaler()  #scaling so all the features are thought about equally
X_scaled = scaler.fit_transform(X)


##  ELBOW METHOD TO SEE HOW MANY CLUSTERS ARE OPTIMAL  (PREVIOUS RESULTS SUGGEST K=2 ) Very slow, does kmeans 10 times
inertias = []
for k in range(1, 10):
    km = KMeans(n_clusters=k)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.plot(range(1, 10), inertias, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

while True:
    try:
        k = int(input("Enter the number of clusters (e.g., 2, 3...): "))
        if k > 0:
            break
        else:
            print("Please enter a positive number.")
    except ValueError:
        print("Invalid input. Please enter a number.")


kmean = KMeans(n_clusters=k, random_state=42)  #kmeans is the simplest, works well with the dataset, and we have learned it in class
kmean.fit(X_scaled)
dataset['Cluster'] = kmean.labels_
print("Model training complete.")

# Display statistical summary
print("\n--- Statistical Summary for Each Cluster ---")
print(dataset.groupby('Cluster')[features].describe().T)

# Display graphical summary (Boxplots)
print("\nGenerating boxplots for each feature by cluster...")
for feature in features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=dataset, x='Cluster', y=feature, palette='Set2')
    plt.title(f'{feature} Distribution by Cluster')
    plt.show()


