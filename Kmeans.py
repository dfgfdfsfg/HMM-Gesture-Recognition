from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from kneed import KneeLocator
from tqdm import tqdm
M_dict = {}
# 假设你有一个名为df的DataFrame，它包含了六列特征
def Kmeans_(filename, M=20):
    features = []
    with open(filename, 'r', encoding='utf8') as f:
        for line in f:
            data = line.split()
            features.append([float(x) for x in data[1:]])

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Use Elbow method to determine the best number of clusters
    sse = []
    k_values = range(1, 48)  # Typically 1 to 10 clusters
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features_scaled)
        sse.append(kmeans.inertia_)  # SSE to elbow plot

    # Plotting the results of the Elbow method
    kneedle = KneeLocator(k_values, sse, curve='convex', direction='decreasing')
    #return kneedle.elbow
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, sse, marker='o')
    plt.plot(kneedle.elbow, sse[kneedle.elbow], marker='x')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters k')
    plt.ylabel('Sum of squared distances (SSE)')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()


#filedir = './train'
Gesturelist = ['beat3', 'beat4', 'circle', 'eight', 'inf', 'wave']
# for name in tqdm(Gesturelist):
#     filedir_ = os.path.join(filedir,name)
#     for txt_file in tqdm(os.listdir(filedir_)):
#         if txt_file.endswith("txt"):
#             filepath = os.path.join(filedir_, txt_file)
#             ans = Kmeans_(filepath)
#             M_dict[ans] = M_dict.get(ans, 0) + 1
# M_sort = sorted(M_dict.items(), key=lambda x: (x[1], x[0]), reverse=True)
# filename = "M_train.txt"
# with open(filename,'w') as f:
#     f.write(" ".join(str(item) for item in M_sort[:5]))
filedir_local = './train_local'
for name in tqdm(Gesturelist):
    filedir_ = os.path.join(filedir_local,name)
    for txt_file in tqdm(os.listdir(filedir_)):
        if txt_file.endswith("txt"):
            filepath = os.path.join(filedir_, txt_file)
            ans = Kmeans_(filepath)
            M_dict[ans] = M_dict.get(ans, 0) + 1
M_sort = sorted(M_dict.items(), key=lambda x: (x[1], x[0]), reverse=True)
filename = "M_train_local.txt"
with open(filename,'w') as f:
    f.write(" ".join(str(item) for item in M_sort[:5]))
