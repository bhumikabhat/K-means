🛒 Customer Segmentation using K-Means Clustering
📌 Objective

This project applies K-Means Clustering on the Mall Customers dataset to group customers into meaningful segments based on their Annual Income and Spending Score.

Clustering helps businesses understand customer behavior and design targeted marketing strategies.

⚙️ Tools & Libraries

Python 3.x

Pandas → Data handling

Matplotlib → Visualization

Scikit-learn → K-Means, Silhouette Score

📊 Steps in the Project

Load Dataset → Read Mall_Customers.csv and explore.

Feature Selection → Use relevant columns (Annual Income, Spending Score).

Data Scaling → Standardize features for better clustering performance.

Elbow Method → Determine the optimal number of clusters (k).

K-Means Clustering → Apply K-Means and assign labels.

Visualization → Plot clusters and centroids.

Evaluation → Compute Silhouette Score to measure cluster quality.

📈 Results

The Elbow Method suggests the best number of clusters (commonly k=5 for this dataset).

Customers are grouped into distinct segments (e.g., high spenders, low spenders, average income groups).

Silhouette Score indicates how well the clusters are separated.

🚀 How to Run the Project

Clone this repository or download the files.

Install dependencies:

pip install pandas matplotlib scikit-learn


Run the script:

python clustering_kmeans.py

📂 Dataset

File: Mall_Customers.csv

Contains customer details:

CustomerID

Gender

Age

Annual Income (k$)

Spending Score (1–100)

📌 Example Visualization

The output shows customer groups with distinct colors and centroids (black X):

Cluster 1: High income, high spending

Cluster 2: Low income, low spending

Cluster 3: Average customers

… etc.

✅ Evaluation Metric

Silhouette Score is used to evaluate clustering performance.

Range: -1 to +1 (closer to 1 means well-clustered).
