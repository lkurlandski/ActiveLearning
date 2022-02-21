"""Acquire a scikit-learn estimator from a string code.
"""

import sklearn
from sklearn.ensemble import RandomForestClassifier     # multiclass
from sklearn.multiclass import OneVsRestClassifier      # ova
from sklearn.neural_network import MLPClassifier        # multiclass
from sklearn.svm import SVC                             # ovo

# TODO: implement system to recieve model kwargs and output custom model
def get_estimator_from_code(code:str) -> sklearn.base.BaseEstimator:
    """Return an instance of a estimator.

    Parameters
    ----------
    code : str
        A code describing which model should be returned

    Returns
    -------
    sklearn.base.BaseEstimator
        A scikit-learn compatible classifier as demanded by ModAL

    Raises
    ------
    ValueError
        If the code is not recognized
    """

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
