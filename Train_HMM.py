from hmmlearn import hmm
import warnings
import os
from get_features import *
from tqdm import tqdm
warnings.filterwarnings("ignore")

Gesturelist = ['beat3', 'beat4', 'circle', 'eight', 'inf', 'wave']


def train(train_dir, n_components=8, n_iter=1800, hmm_cls=hmm.GaussianHMM):
    # =============补充===============
    models = {}
    for name in tqdm(Gesturelist):
        digit_dir = os.path.join(train_dir, name)
        features = []
        for txt_file in os.listdir(digit_dir):
            if txt_file.endswith('.txt'):
                path = os.path.join(digit_dir, txt_file)
                g_feat = gcff(path)
                features.append(g_feat)
        features = np.concatenate(features, axis=0)
        model = hmm_cls(n_components=n_components, covariance_type="diag", n_iter=n_iter)
        model.fit(features)
        models[name] = model
    return models
