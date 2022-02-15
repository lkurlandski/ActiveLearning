"""Get training and test data.
"""

from pathlib import Path

import numpy as np

from sklearn.datasets import fetch_20newsgroups, fetch_20newsgroups_vectorized, fetch_covtype
from sklearn.model_selection import train_test_split

def shuffle_corresponding_arrays(a1, a2, random_state):
    
    if a1.shape[0] != a2.shape[0]:
        raise ValueError("Arrays are different lengths along first axis.")
    
    rng = np.random.default_rng(random_state)
    idx = np.arange(a1.shape[0])
    rng.shuffle(idx)
    
    return a1[idx], a2[idx]

def get_covtype(random_state):
    
    bunch = fetch_covtype(random_state=random_state, shuffle=True)
    
    X = bunch['data']
    y = bunch['target']
    labels = [1,2,3,4,5,6,7]    # bunch['target_names'] appears to be broken...
    
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.25, random_state=random_state)
        
    return X_train, X_test, y_train, y_test, labels

# TODO: convert to the scipy.sparse.csr_array type instead of scipy.sparse.csr_matrix,
    # then ensure the rest of the system is compatible with this type.
def get_20_newsgroups(random_state):
    
    bunch = fetch_20newsgroups_vectorized(
        subset='train', remove=('headers', 'footers', 'quotes'), random_state=random_state
    )
    X_train = bunch['data']
    y_train = bunch['target']
    labels = bunch['target_names']
    
    bunch = fetch_20newsgroups_vectorized(
        subset='test', remove=('headers', 'footers', 'quotes'), random_state=random_state
    )
    X_test = bunch['data']
    y_test = bunch['target']
    labels = bunch['target_names']
        
    return X_train, X_test, y_train, y_test, labels

# TODO: attempt to access bunch['file_names'] and create a streaming approach!
def get_20_newsgroups_bert(random_state):
    
    bunch = fetch_20newsgroups(
        subset='train', remove=('headers', 'footers', 'quotes'), random_state=random_state
    )
    X_train = np.array(bunch['data'])
    y_train = np.array(bunch['target'])
    labels = bunch['target_names']
    
    bunch = fetch_20newsgroups(
        subset='test', remove=('headers', 'footers', 'quotes'), random_state=random_state
    )
    X_test = np.array(bunch['data'])
    y_test = np.array(bunch['target'])
    labels = bunch['target_names']
        
    return X_train, X_test, y_train, y_test, labels

def get_avila(random_state):
    
    avila_root = Path("/projects/nlp-ml/io/input/numeric/Avila")
    
    X_train = np.loadtxt(avila_root / "train/X.csv", delimiter=',')
    y_train = np.loadtxt(avila_root / "train/y.csv", dtype=np.str_)
    X_test = np.loadtxt(avila_root / "test/X.csv", delimiter=',')
    y_test = np.loadtxt(avila_root / "test/y.csv", dtype=np.str_)
    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "W", "X", "Y"]
    
    if random_state is not None:
        X_train, y_train = shuffle_corresponding_arrays(X_train, y_train, random_state)
        X_test, y_test = shuffle_corresponding_arrays(X_train, y_train, random_state)
    
    return X_train, X_test, y_train, y_test, labels

def get_dataset(dataset, random_state=None):
    if dataset == "Avila":
        return get_avila(random_state)
    elif dataset == "20NewsGroups":
        return get_20_newsgroups(random_state)
    elif dataset == "20NewsGroups-raw":
        return get_20_newsgroups_bert(random_state)
    elif dataset == "Covertype":
        return get_covtype(random_state)
    else:
        raise ValueError(f"Dataset not recognized: {dataset}")

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, labels = get_dataset("20NewsGroups-bert", 0)
    print()