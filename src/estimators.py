"""Acquire a scikit-learn estimator from a string code.

This module is heavily reliant upon scikit-learn. Users are advised to refer to the following docs:
    1.12. Multiclass and multioutput algorithms 
        https://scikit-learn.org/stable/modules/multiclass.html
    
"""
from dataclasses import dataclass
import inspect
from typing import Any, Callable, Dict, Set

import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV 
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

# sklearn uses two different strings to encode the different types of multiclass modes
sklearn_multi_class_code_map = {
    'one_versus_rest' : 'ovr',
    'one_versus_one' : 'ovo',
    'ovr' : 'ovr',
    'ovo' : 'ovo',
    'occ' : 'occ',
    'crammer_singer' : 'crammer_singer',
}

@dataclass
class BaseLearner:
    learner: Callable[..., sklearn.base.BaseEstimator]
    kwargs: Dict[str, Any]
    default_multiclass_code: str
    multi_class_parameter_options: Set[str]
    
@dataclass
class MultiClassEstimator:
    wrapper: Callable[..., sklearn.base.BaseEstimator]
    kwargs: Dict[str, Any]
    
@dataclass
class MultiOutputEstimator:
    wrapper: Callable[..., sklearn.base.BaseEstimator]
    kwargs: Dict[str, Any]

def probabalistic_estimator(estimator):
    if hasattr(estimator, 'predict_proba'):
        return estimator
    return CalibratedClassifierCV(estimator)

# TODO: implement system to recieve model kwargs and output custom model
# TODO: implement multioutput mode
# TODO: implement multitask mode, which is a combination of multiclass and multioutput
def get_estimator_from_code(
        learner_code: str,
        multiclass_code: str = None,
        multioutput_code:str = None,
        target_names: np.ndarray = None,
    ) -> sklearn.base.BaseEstimator:

    learner_map = {
        "mlp" : BaseLearner(MLPClassifier, {'hidden_layer_sizes' : (64,)}, 'inherint', set()),
        "svm" : BaseLearner(LinearSVC, {}, 'ovr', set('ovr', 'crammer_singer')),
        "rf" : BaseLearner(RandomForestClassifier, {}, 'inherint', set()),
    }
    
    multiclass_map = {
        "inherint" : MultiClassEstimator(lambda x: x, {}),
        "ovr" : MultiClassEstimator(OneVsRestClassifier, {'n_jobs': -1}),    
        "ovo" : MultiClassEstimator(OneVsOneClassifier, {'n_jobs': -1}),
        "occ" : MultiClassEstimator(OutputCodeClassifier, {"random_state": None, "n_jobs": None}),
    }
    
    multioutput_map = {
        ""
    }

    if learner_code not in learner_map:
        raise ValueError(f"Learner code not recognized: {learner_code}")
    if multiclass_code == 'crammer_singer' and not isinstance(learner_map[learner_code].learner, LinearSVC):
        raise ValueError(f"crammer_singer multiclass mode is only implemented for LinearSVC")
    elif multiclass_code not in multiclass_map:
        raise ValueError(f"Multiclass code not recognized: {multiclass_code}")
    
    bl = learner_map[learner_code]
    ml = multiclass_map[multiclass_code]
    
    # Handle the case where the default multiclassification mode is the desired one, or binary case
    if bl.default_multiclass_code == multiclass_code or len(target_names) <= 2:
        estimator = bl.learner(bl.kwargs)
        estimator = probabalistic_estimator(estimator)
        return estimator

    # Handle the case where the default multiclass mode not desired
    learner_argspec = inspect.getfullargspec(bl.learner)
    learner_args = set(learner_argspec.args + learner_argspec.kwonlyargs)
    if 'multi_class' in learner_args:
        bl.kwargs['multi_class'] = sklearn_multi_class_code_map[multiclass_code]
        estimator = bl.learner(bl.kwargs)
        estimator = probabalistic_estimator(estimator)
        return estimator
    
    # Handle the case where the default multiclass mode not desired and wrapper required
    estimator = bl.learner(bl.kwargs)
    estimator = probabalistic_estimator(estimator)
    estimator = ml.wrapper(estimator, ml.kwargs)
    return estimator
