from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
from collections import Counter
from LH_HMM import LH_HMM
import os
import pickle
import numpy as np
from hmmlearn import hmm
app = Flask(__name__)
Gesturelist = ['beat3', 'beat4', 'circle', 'eight', 'inf', 'wave']
model = LH_HMM(hmm_cls=1, N=10, M=30, delta=2, flags=False)

with open('hmm_model_params_de.pkl', 'rb') as file:
    params = pickle.load(file)
# 设置模型参数
for name in Gesturelist:
    covars = params['covars'][name]
    if covars.ndim == 3:
        # 提取对角线元素
        diag_covars = np.array([np.diag(covars[i]) for i in range(covars.shape[0])])
        params['covars'][name] = diag_covars
for name in Gesturelist:
    model.models[name] = hmm.GaussianHMM(n_components=model.N, covariance_type="diag", n_iter=model.n_iter)
    model.models[name].transmat_ = params['transmat'][name]
    model.models[name].means_ = params['means'][name]
    model.models[name].covars_ = params['covars'][name]
    model.models[name].startprob_ = params['startprob'][name]
cnt = 200
@app.route('/recognition', methods=['POST'])
def calculate_similarity():
    print("received data from the web.")
    data = request.get_json()
    global cnt
    # 创建文件夹（如果它不存在的话）
    folder_path = 'test_for_android'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, f'{cnt}.txt')

    with open(file_path, 'w') as file:
        for data_ in data:
            for key, value in data_.items():
                file.write(value + ' ')
            file.write('\n')
    cnt += 1
    x_test = model.dataloder_android(file_path)
    y_pred = model.predict(x_test)
    print(y_pred)

    return jsonify(y_pred)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
