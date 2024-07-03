import warnings
import os
from get_features import *

warnings.filterwarnings("ignore")

Gesturelist = ['beat3', 'beat4', 'circle', 'eight', 'inf', 'wave']


def test(test_dir, models):
    # =============补充===============
    correct = 0
    total = 0
    pred = []
    truth = []
    for name in Gesturelist:
        digit_dir = os.path.join(test_dir, name)
        features = []
        for txt_file in os.listdir(digit_dir):
            if txt_file.endswith('.txt'):
                path = os.path.join(digit_dir, txt_file)
                g_feat = gcff(path)
                features.append(g_feat)
                log_likelihood = {model_name: model.score(g_feat) for model_name, model in models.items()}
                predicted_name = max(log_likelihood, key=log_likelihood.get)
                correct += (predicted_name == name)
                pred.append(predicted_name)
                truth.append(name)
                total += 1
    acc = correct / total if total > 0 else 0
    return acc, truth, pred


def Test(test_dir, models):
    features = []
    predicted_name = []
    ture_name = []
    for txt_file in os.listdir(test_dir):
        if txt_file.endswith('.txt'):
            path = os.path.join(test_dir, txt_file)
            g_feat = gcff(path)
            features.append(g_feat)
            log_likelihood = {model_name: model.score(g_feat) for model_name, model in models.items()}
            predicted_name.append(max(log_likelihood, key=log_likelihood.get))
            ture_name.append(txt_file)
    return predicted_name,ture_name
