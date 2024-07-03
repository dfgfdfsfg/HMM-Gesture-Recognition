from hmmlearn import hmm
# from Train_HMM import *
# from Test_HMM import *
from sklearn.model_selection import StratifiedKFold

from LH_HMM import LH_HMM
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle
# models = train('./train', n_components=8)
# acc, acc1, acc2 = test('./test_local', models)
# print(f"acc = {acc}")
# print(f"True:{acc1}")
# print(f"pred:{acc2}")
# models = LH_HMM(N=12, M=8, delta=0, flags=False, hmm_cls=1)
# X, Y = models.dataloder('./train_local')
# models.fit(X, Y)
# x_test, y_test = models.dataloder('./test_local')
# y_pred = models.predict(x_test)
# report = classification_report(y_test,y_pred)
# print(report)
import hmmlearn

print(hmmlearn.__version__)
from hmmlearn import hmm
import numpy as np

Gesturelist = ['beat3', 'beat4', 'circle', 'eight', 'inf', 'wave']
model = LH_HMM(hmm_cls=1, N=10, M=48, delta=1, flags=False)
X_train, Y_train = model.dataloder('./train_local')
model.fit(X_train, Y_train)
X_test, Y_test = model.dataloder('./test_local')
Y_pred = model.predict(X_test)
report = classification_report(Y_test, Y_pred)
print(report)

# transmat = {}
# means = {}
# covars = {}
# startprob = {}
# for name in Gesturelist:
#     transmat[name] = model.models[name].transmat_
#     means[name] = model.models[name].means_
#     covars[name] = model.models[name].covars_
#     print(model.models[name].covars_.shape)
#     startprob[name] = model.models[name].startprob_
# print(transmat.keys())
# print(means.keys())
# print(covars.keys())
# print(startprob.keys())
# #使用 pickle 保存模型参数
# with open('hmm_model_params_de.pkl', 'wb') as file:
#     pickle.dump({
#         'transmat': transmat,
#         'means': means,
#         'covars': covars,
#         'startprob': startprob
#     }, file)

# model_ = LH_HMM(hmm_cls=1, N=10, M=30, delta=1, flags=False)
#
# with open('hmm_model_params.pkl', 'rb') as file:
#     params = pickle.load(file)
# # 设置模型参数
# for name in Gesturelist:
#     covars = params['covars'][name]
#     if covars.ndim == 3:
#         # 提取对角线元素
#         diag_covars = np.array([np.diag(covars[i]) for i in range(covars.shape[0])])
#         params['covars'][name] = diag_covars
# for name in Gesturelist:
#     model_.models[name] = hmm.GaussianHMM(n_components=model_.N, covariance_type="diag", n_iter=model_.n_iter)
#     model_.models[name].transmat_ = params['transmat'][name]
#     model_.models[name].means_ = params['means'][name]
#     model_.models[name].covars_ = params['covars'][name]
#     model_.models[name].startprob_ = params['startprob'][name]
# x_test,y_test = model_.dataloder('./test_local')
# y_pred_ = model_.predict(x_test)
# report = classification_report(y_test, y_pred_)
# print(report)
# '''交叉验证'''
# param_grid = {
#     'N': [8, 10, 12],  # HMM隐藏状态数
#     'M': [7, 8, 9, 10, 11],  # kmeans聚类数
#     'delta': [0, 1, 2],  # 特征差分的级别
#     'flags': [True, False]  # 返回聚类中心或每个聚类的平均值
# }
# file_name_ = ["Hmm.GaussianHMM_train.txt", "Hmm.GMMHMM_train.txt", "Hmm.GaussianHMM_train_local.txt",
#               "Hmm.GMMHMM_train_local.txt"]
# train_ = ['./train', './train', './train_local', './train_local']
# test_ = ['./test', './test', './test_local', './test_local']
# hmm_cls_ = [1, 0, 1, 0]
# n_splits_ = [4, 4, 30, 30]
# for i in range(4):
#     model = LH_HMM(hmm_cls=hmm_cls_[i])
#     x_train, y_train = model.dataloder(train_[i])
#     skf = StratifiedKFold(n_splits=n_splits_[i])
#     grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=skf, scoring='accuracy', verbose=1, n_jobs=-1)
#     grid_search.fit(x_train, y_train)
#     best_model = grid_search.best_estimator_
#     file_name = "kalman_" + file_name_[i]
#     with open(file_name, 'w') as f:
#         print("Best parameters:", grid_search.best_params_)
#         print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
#         x_test, y_test = model.dataloder(test_[i])
#         y_pred = best_model.predict(x_test)
#         report = classification_report(y_test, y_pred)
#         print(report)
#         f.write(f"Best parameters:{grid_search.best_params_}")
#         f.write('\n')
#         f.write("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
#         f.write('\n')
#         f.write(report)
#     transmat = {}
#     means = {}
#     covars = {}
#     startprob = {}
#     for name in Gesturelist:
#         transmat[name] = best_model.models[name].transmat_
#         means[name] = best_model.models[name].means_
#         covars[name] = best_model.models[name].covars_
#         startprob[name] = best_model.models[name].startprob_
#     with open(f'hmm_model_params_de_{i}.pkl', 'wb') as file:
#         pickle.dump({
#             'transmat': transmat,
#             'means': means,
#             'covars': covars,
#             'startprob': startprob
#         }, file)

# '''验证隐藏状态数的影响'''
# filename = 'different_N.txt'
# with open(filename, 'w') as f:
#     model1 = LH_HMM(N=12, M=11, delta=2, flags=True, hmm_cls=1)
#     x_train1, y_train1 = model1.dataloder('train')
#     model1.fit(x_train1, y_train1)
#     x_test1, y_test1 = model1.dataloder('test')
#     y_pred1 = model1.predict(x_test1)
#     report1 = classification_report(y_test1, y_pred1)
#     f.write("Hmm.GaussianHMM_train with N = 12\n")
#     f.write(report1)
#     f.write("\n================================================================\n")
#     model2 = LH_HMM(N=8, delta=0, M=8, flags=False, hmm_cls=1)
#     x_train2, y_train2 = model2.dataloder('train_local')
#     model2.fit(x_train2, y_train2)
#     x_test2, y_test2 = model2.dataloder('test_local')
#     y_pred2 = model2.predict(x_test2)
#     report2 = classification_report(y_test2, y_pred2)
#     f.write("Hmm.GaussianHMM_train_local with N = 8\n")
#     f.write(report2)
#     f.write("\n================================================================\n")

# '''验证聚类数的影响'''
# M = [7, 8, 9, 10, 11]
# filename = 'different_M_Hmm.GaussianHMM_train_local.txt'
# with open(filename, 'w') as f:
#     for i in M:
#         model = LH_HMM(N=12, delta=0, flags=False, hmm_cls=1)
#         x_train, y_train = model.dataloder('./train_local')
#         model.fit(x_train,y_train)
#         x_test, y_test = model.dataloder('./test_local')
#         y_pred = model.predict(x_test)
#         report = classification_report(y_test, y_pred)
#         f.write(f"聚类数M={i}\n")
#         f.write(report)
#         f.write("\n================================================\n")

# '''验证delta'''
# filename = 'different_delta.txt'
# with open(filename, 'w') as f:
#     model1 = LH_HMM(N=8, M=11, delta=0, flags=True, hmm_cls=1)
#     x_train1, y_train1 = model1.dataloder('train')
#     model1.fit(x_train1, y_train1)
#     x_test1, y_test1 = model1.dataloder('test')
#     y_pred1 = model1.predict(x_test1)
#     report1 = classification_report(y_test1, y_pred1)
#     f.write("Hmm.GaussianHMM_train with delta = 0\n")
#     f.write(report1)
#     f.write("\n================================================================\n")
#     model2 = LH_HMM(N=12, delta=2, M=8, flags=False, hmm_cls=1)
#     x_train2, y_train2 = model2.dataloder('train_local')
#     model2.fit(x_train2, y_train2)
#     x_test2, y_test2 = model2.dataloder('test_local')
#     y_pred2 = model2.predict(x_test2)
#     report2 = classification_report(y_test2, y_pred2)
#     f.write("Hmm.GaussianHMM_train_local with delta = 2\n")
#     f.write(report2)
#     f.write("\n================================================================\n")

# '''聚类中心or聚类平均值'''
# filename = 'different_kmeans.txt'
# with open(filename, 'w') as f:
#     model2 = LH_HMM(N=12, delta=0, M=8, flags=True, hmm_cls=1)
#     x_train2, y_train2 = model2.dataloder('train_local')
#     model2.fit(x_train2, y_train2)
#     x_test2, y_test2 = model2.dataloder('test_local')
#     y_pred2 = model2.predict(x_test2)
#     report2 = classification_report(y_test2, y_pred2)
#     f.write("Hmm.GaussianHMM_train_local with flags = True\n")
#     f.write(report2)
#     f.write("\n================================================================\n")
