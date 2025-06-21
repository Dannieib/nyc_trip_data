import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_parquet("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet")

#print(df.head(100))
print(df[["VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count",
          "trip_distance", "fare_amount", "improvement_surcharge",
          "total_amount", "congestion_surcharge"]].head())

df_cluster = df[["trip_distance", "fare_amount", "passenger_count"]].dropna() #Data clean up

scaler = StandardScaler()
X = scaler.fit_transform(df_cluster)
kmeans = KMeans(n_clusters=3, random_state=42)
df_cluster['cluster'] = kmeans.fit_predict(X)

#Plot 2D clusters
plt.scatter(X[:, 0], X[:, 1], c=df_cluster['cluster'], cmap='viridis')
plt.xlabel("Trip Distance (scaled)")
plt.ylabel("Fare Amount (scaled)")
plt.title("NYC Yellow Taxi Trip Clusters")
plt.show()