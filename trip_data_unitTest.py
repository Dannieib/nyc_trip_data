import unittest
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

class TestKMeansTaxi(unittest.TestCase):

    def setUp(self):
        self.df = pd.read_parquet("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet")
        self.df = self.df[['trip_distance', 'fare_amount', 'passenger_count']].dropna()
        self.df = self.df[
            (self.df['trip_distance'] > 0) &
            (self.df['fare_amount'] > 0) &
            (self.df['passenger_count'] > 0)
            ]
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.df)

    def test_cluster_output(self):
        model = KMeans(n_clusters=3, random_state=42)
        labels = model.fit_predict(self.X)
        self.assertEqual(len(labels), len(self.df))

    def test_number_of_clusters(self):
        model = KMeans(n_clusters=3, random_state=42)
        labels = model.fit_predict(self.X)
        unique_labels = set(labels)
        self.assertEqual(len(unique_labels), 3)

    def test_no_nan_in_scaled_data(self):
        nan_count = np.isnan(self.X).sum()
        self.assertEqual(nan_count, 0)

    def test_cluster_centers_shape(self):
        model = KMeans(n_clusters=3, random_state=42)
        model.fit(self.X)
        self.assertEqual(model.cluster_centers_.shape, (3, self.X.shape[1]))

if __name__ == '__main__':
    unittest.main()
