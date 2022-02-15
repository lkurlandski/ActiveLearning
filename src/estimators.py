"""Acquire a scikit-learn estimator from a string code.
"""

from sklearn.ensemble import RandomForestClassifier     # multiclass
from sklearn.multiclass import OneVsRestClassifier      # ova
from sklearn.neural_network import MLPClassifier        # multiclass
from sklearn.svm import SVC                             # ovo

# FIXME: implement system to recieve model kwargs and output custom model
def get_estimator_from_code(code, **model_kwargs):
    if code == 'mlp':
        return MLPClassifier()
    elif code == 'svm':
        return SVC(kernel='linear', probability=True)
    elif code == 'svm-ova':
        return OneVsRestClassifier(SVC(kernel='linear', probability=True))
    elif code == 'rf':
        return RandomForestClassifier()
    else:
        raise ValueError(f"code: {code} not recognized.")