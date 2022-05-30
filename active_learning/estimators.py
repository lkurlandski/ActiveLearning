"""Acquire a scikit-learn estimator from a string code.
"""


from pprint import pformat, pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from typing import Union

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier


# Keeping this dictionary up to date can help with debugging and error messages.
valid_base_learners = {
    "ExtraTreesClassifier",
    "GradientBoostingClassifier",
    "RandomForestClassifier",
    "SGDClassifier",
    "KNeighborsClassifier",
    "RadiusNeighborsClassifier",
    "MLPClassifier",
    "LinearSVC",
    "SVC",
    "DecisionTreeClassifier",
    "ExtraTreeClassifier",
}


# These learners are multiclass by nature and should not be wrapped in a
# multiclass wrapper.
inherintly_multiclass = {
    "BernoulliNB",
    "DecisionTreeClassifier",
    "ExtraTreeClassifier",
    "ExtraTreesClassifier",
    "GaussianNB",
    "KNeighborsClassifier",
    "LabelPropagation",
    "LabelSpreading",
    "LinearDiscriminantAnalysis",
    "LinearSVC",  # (setting multi_class=”crammer_singer”)
    "LogisticRegression",  # (setting multi_class=”multinomial”)
    "LogisticRegressionCV",  # (setting multi_class=”multinomial”)
    "MLPClassifier",
    "NearestCentroid",
    "QuadraticDiscriminantAnalysis",
    "RadiusNeighborsClassifier",
    "RandomForestClassifier",
    "RidgeClassifier",
    "RidgeClassifierCV",
}


# These learners are not inherintly multiclass, but support multiclass
# classification using a one-versus-one protocol.
multiclass_as_one_vs_one = {
    "NuSVC",
    "SVC",
    "GaussianProcessClassifier",  # (setting multi_class = “one_vs_one”)
}


# These learners are not inherintly multiclass, but support multiclass
# classification using a one-versus-rest protocol.
multiclass_as_one_vs_rest = {
    "GradientBoostingClassifier",
    "GaussianProcessClassifier",  # (setting multi_class = “one_vs_rest”)
    "LinearSVC",  # (setting multi_class=”ovr”)
    "LogisticRegression",  # (setting multi_class=”ovr”)
    "LogisticRegressionCV",  # (setting multi_class=”ovr”)
    "SGDClassifier",
    "Perceptron",
    "PassiveAggressiveClassifier",
}


# These learners support multilabel classification and should not be wrapped in
# a multiclass or multilabel wrapper.
support_multilabel = {
    "DecisionTreeClassifier",
    "ExtraTreeClassifier",
    "ExtraTreesClassifier",
    "KNeighborsClassifier",
    "MLPClassifier",
    "RadiusNeighborsClassifier",
    "RandomForestClassifier",
    "RidgeClassifier",
    "RidgeClassifierCV",
}


# These learners support multioutput classification and need their prediction
# targets modified to be compatible with ModAL.
# The MultiOutputToMultiLabelClassifier handles this.
support_multiclass_multioutput = {
    "DecisionTreeClassifier",
    "ExtraTreeClassifier",
    "ExtraTreesClassifier",
    "RandomForestClassifier",
    "KNeighborsClassifier",
    "RadiusNeighborsClassifier",
}


# Some scikit-learn estimators do not support sparse target vectors and must be
# converted to dense.
sparse_multilabel_indicator_not_supported = {"RandomForestClassifier"}


class MultiOutputToMultiLabelClassifier:
    """Change the shape of the predictions of a multi-output classifier."""

    def __init__(self, clf: BaseEstimator) -> None:
        """Instantiate the wrapper around a scikit-learn classifier.

        Parameters
        ----------
        clf : BaseEstimator
            The classifier to wrap.
        """
        self.clf = clf

    def __repr__(self):
        return "MultiOutputToMultiLabelClassifier(" + self.clf.__repr__() + ")"

    def __str__(self):
        return "MultiOutputToMultiLabelClassifier(" + self.clf.__repr__() + ")"

    def predict_proba(self, X: Union[np.ndarray, sparse.csr_matrix]) -> np.ndarray:
        """Predict class probabilities for X.

        Parameters
        ----------
        X : Union[np.ndarray, sparse.csr_matrix]
            The input samples.

        Returns
        -------
        np.ndarray
            The class probabilities of the input samples.

        Raises
        ------
        ValueError
            If the target shape of predict_proba is not (n_classes, n_samples, 2).
        """
        probas = self.clf.predict_proba(X)
        if not all(p.shape == (X.shape[0], 2) for p in probas):
            raise ValueError(
                f"Expected every individual probability to have shape {(X.shape[0], 2)},"
                f" but predict_proba() returned shape {np.array(probas).shape}. "
                "This likely indicates that MultiOutputToMultiLabelClassifier should not be used."
            )
        probas = [p[:, 1] for p in probas]
        probas = np.vstack(probas).T
        return probas

    def predict(self, X: Union[np.ndarray, sparse.csr_matrix]) -> np.ndarray:
        """Predict classes for X.

        Parameters
        ----------
        X : Union[np.ndarray, sparse.csr_matrix]
            The input samples.

        Returns
        -------
        np.ndarray
            The predicted classes.
        """
        return self.clf.predict(X)

    def fit(
        self, X: Union[np.ndarray, sparse.csr_matrix], y: Union[np.ndarray, sparse.csr_matrix]
    ) -> BaseEstimator:
        """Fit the classifier on training data.

        Parameters
        ----------
        X : Union[np.ndarray, sparse.csr_matrix]
            The input samples.
        y : Union[np.ndarray, sparse.csr_matrix]
            The corresponding target values.

        Returns
        -------
        BaseEstimator
            A fitted instance of self.
        """
        self.clf.fit(X, y)
        return self


# TODO: there is no support for determining this information a priori, so we
# should continually update this method as needed to account for bugs we find.
def convert_to_valid_X_y(
    learner: Union[str, BaseEstimator, MultiOutputToMultiLabelClassifier],
    X: Union[np.ndarray, sparse.csr_matrix],  # pylint: disable=unused-argument
    y: Union[np.ndarray, sparse.csr_matrix],
) -> Union[np.ndarray, sparse.csr_matrix]:
    """Convert training data into valid data structures for scikit-learn learners.

    Parameters
    ----------
    learner : Union[str, BaseEstimator, MultiOutputToMultiLabelClassifier]
        The learner to use.
    X : Union[np.ndarray, sparse.csr_matrix]
        Input samples.
    y : Union[np.ndarray, sparse.csr_matrix]
        The corresponding target values.

    Returns
    -------
    Union[np.ndarray, sparse.csr_matrix]
        Input samples and corresponding target values in a possibly different
        data structure.
    """
    if isinstance(learner, MultiOutputToMultiLabelClassifier):
        learner = learner.clf.__class__.__name__
    if isinstance(learner, BaseEstimator):
        learner = learner.__class__.__name__

    if learner in sparse_multilabel_indicator_not_supported and sparse.issparse(y):
        y = y.toarray()

    return X, y


def get_estimator(
    learner: str,
    target_type: str,
    random_state: int,
) -> BaseEstimator:
    """Get a scikit learn estimator given a desired base learner and a multiclass protocol.

    Parameters
    ----------
    learner : str
        Code to refer to the base learner to use.
    target_type : str
        Type of target vector for classification, as returned by
            sklearn.utils.multiclass.type_of_target(). Currently supported targets are one of:
            {'multilabel-indicator', 'multiclass-multioutput', 'multiclass', 'binary'}

    Returns
    -------
    BaseEstimator
        A scikit learn estimator that can be used for multiclass classification.

    Raises
    ------
    KeyError
        If the learner is not recognized.
    """
    # Create the base classification object
    if learner == "DecisionTreeClassifier":
        clf = DecisionTreeClassifier(random_state=random_state)
    elif learner == "ExtraTreeClassifier":
        clf = ExtraTreeClassifier(random_state=random_state)
    elif learner == "ExtraTreesClassifier":
        clf = ExtraTreesClassifier(random_state=random_state)
    elif learner == "RandomForestClassifier":
        clf = RandomForestClassifier(random_state=random_state)
    elif learner == "KNeighborsClassifier":
        clf = KNeighborsClassifier()
    elif learner == "RadiusNeighborsClassifier":
        clf = RadiusNeighborsClassifier(random_state=random_state)
    elif learner == "GradientBoostingClassifier":
        clf = GradientBoostingClassifier(random_state=random_state)
    elif learner == "MLPClassifier":
        clf = MLPClassifier(random_state=random_state)
    elif learner == "LinearSVC":
        clf = LinearSVC(random_state=random_state)
    elif learner == "SVC":
        clf = SVC(kernel="linear", probability=True, random_state=random_state)
    elif learner == "SGDClassifier":
        clf = SGDClassifier(random_state=random_state, n_jobs=-1)
    else:
        raise ValueError(
            f"Learner not recognized: {learner}. "
            f"Valid learners are: {pformat(valid_base_learners)}."
        )

    # Applies a multiclass or multilabel wrapper, if nessecary
    if target_type == "binary":
        return clf
    if target_type == "multiclass":
        if learner in inherintly_multiclass:
            return clf
        return OneVsRestClassifier(clf, n_jobs=-1)
    if target_type == "multilabel-indicator":
        if learner in support_multiclass_multioutput:
            return MultiOutputToMultiLabelClassifier(clf)
        if learner in support_multilabel:
            return clf
        return OneVsRestClassifier(clf, n_jobs=-1)

    raise ValueError(
        f"Estimators for this kind of target: {target_type} not supported. "
        f"Valid target types are:\n{pformat({'multilabel-indicator', 'multiclass', 'binary'})}"
    )
