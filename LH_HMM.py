from hmmlearn import hmm
import warnings
import os
from tqdm import tqdm
import numpy as np

warnings.filterwarnings("ignore")
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class LH_HMM(BaseEstimator, ClassifierMixin):
    def __init__(self, N=10, M=15, delta=2, flags=True, hmm_cls=0):
        super().__init__()
        self.N = N
        self.M = M
        self.delta = delta
        self.flags = flags
        self.n_iter = 1800
        self.hmm_cls = hmm_cls
        self.k = -600
        self.Gesturelist = ['beat3', 'beat4', 'circle', 'eight', 'inf', 'wave']
        self.models = {}

    def dataloder(self, data_path):
        x_list = []
        y_list = []
        for name in self.Gesturelist:
            digit_dir = os.path.join(data_path, name + ".kalman")

            features = []
            for txt_file in os.listdir(digit_dir):
                if txt_file.endswith('.txt'):
                    path = os.path.join(digit_dir, txt_file)
                    g_feat = self.gcff(path)
                    features.append(g_feat)
                    x_list.append(g_feat)
                    y_list.append(name)
        x = np.array(x_list)
        y = np.array(y_list)
        return x, y

    def dataloder_android(self, data_path):
        features = []
        features.append(self.gcff(data_path))
        features = np.array(features)
        return features

    def fit(self, X, Y):
        self.models = {}
        for name in tqdm(self.Gesturelist):
            features = [X[i] for i, label in enumerate(Y) if label == name]
            if features:
                features = np.concatenate(features, axis=0)
                if self.hmm_cls == 0:
                    model = hmm.GMMHMM(n_components=self.N, covariance_type="diag", n_iter=self.n_iter)
                    model.fit(features)
                    self.models[name] = model
                else:
                    model = hmm.GaussianHMM(n_components=self.N, covariance_type="diag", n_iter=self.n_iter)
                    model.fit(features)
                    self.models[name] = model

    def predict(self, X):
        predictions = []
        #max_log_likelihood = []
        for features in X:
            log_likelihood = {model_name: model.score(features) for model_name, model in self.models.items()}
            predicted_name = max(log_likelihood, key=log_likelihood.get)
            if max(log_likelihood.values())<=self.k:
                predicted_name = "null"
            predictions.append(predicted_name)

            #max_log_likelihood.append(max(log_likelihood.values()))
       # return predictions, max_log_likelihood
        return predictions


    def score(self, X, Y):
        total = 0
        correct = 0
        predictions = self.predict(X)
        print(predictions)
        print(Y)
        for name, predicted_name in zip(Y, predictions):
            correct += (name == predicted_name)
            total += 1
        return correct / total if total > 0 else 0

    def kmeans_(self, path):
        features = []
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                data = line.split()
                features.append([float(x) for x in data[1:]])

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=self.M, random_state=42)
        kmeans.fit(features_scaled)

        if self.flags:
            cluster_centers = kmeans.cluster_centers_
            cluster_centers_original_scale = scaler.inverse_transform(cluster_centers)
            return cluster_centers_original_scale
        else:
            cluster_labels = kmeans.labels_
            cluster_averages = np.zeros_like(kmeans.cluster_centers_)
            for i in range(self.M):
                cluster_averages[i] = np.mean(features_scaled[cluster_labels == i], axis=0)
            cluster_averages_original_scale = scaler.inverse_transform(cluster_averages)
            return cluster_averages_original_scale

    def gcff(self, path):
        cluster_centers = self.kmeans_(path)
        data = cluster_centers.T
        ans = [data]
        ans1 = []
        diff_1 = []
        diff_2 = []
        if self.delta >= 1:
            for line in data:
                diff = np.diff(line, prepend=line[0])
                diff_1.append(diff)
            ans.append(diff_1)
            ans1.append(diff_1)
        if self.delta >= 2:
            for line in diff_1:
                diff = np.diff(line, prepend=line[0])
                diff_2.append(diff)
            ans.append(diff_2)
            ans1.append(diff_2)
        # print(ans)
        return np.transpose(np.concatenate(ans1, axis=0), [1, 0])
