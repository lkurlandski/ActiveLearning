"""Get training and test data.
"""

from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups, fetch_20newsgroups_vectorized, fetch_covtype, load_iris
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
    labels = [1,2,3,4,5,6,7]    # bunch['target_names'] does not work
    
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.25, random_state=random_state)
        
    return X_train, X_test, y_train, y_test, labels

# TODO: use scipy.sparse.csr_array not scipy.sparse.csr_matrix (per the scipy.sparse docs)
def get_20_newsgroups(random_state):
    
    bunch = fetch_20newsgroups_vectorized(
        subset='train', remove=('headers', 'footers', 'quotes')
    )
    X_train = bunch['data']
    y_train = bunch['target']
    X_train, y_train = shuffle_corresponding_arrays(X_train, y_train, random_state)
    
    bunch = fetch_20newsgroups_vectorized(
        subset='test', remove=('headers', 'footers', 'quotes')
    )
    X_test = bunch['data']
    y_test = bunch['target']
    labels = list(bunch['target_names'])
        
    return X_train, X_test, y_train, y_test, labels

def get_20_newsgroups_bert(random_state):
    
    bunch = fetch_20newsgroups(
        subset='train', remove=('headers', 'footers', 'quotes')
    )
    X_train = np.array(bunch['data'])
    y_train = np.array(bunch['target'])
    X_train, y_train = shuffle_corresponding_arrays(X_train, y_train, random_state)
    
    bunch = fetch_20newsgroups(
        subset='test', remove=('headers', 'footers', 'quotes')
    )
    X_test = np.array(bunch['data'])
    y_test = np.array(bunch['target'])
    labels = list(bunch['target_names'])
        
    return X_train, X_test, y_train, y_test, labels

def get_iris(random_state):
    
    bunch = load_iris()
    X = np.array(bunch['data'])
    y = np.array(bunch['target'])
    labels = bunch['target_names'].tolist()
    
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.25, random_state=random_state)
        
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

# TODO: Establish proper data types and ensure proper data types are being used
# TODO: attempt to access bunch['file_names'] and create a streaming approach for text datasets
def get_dataset(dataset, random_state=None):
    if dataset == "Avila":
        X_train, X_test, y_train, y_test, labels = get_avila(random_state)
    elif dataset == "20NewsGroups":
        X_train, X_test, y_train, y_test, labels = get_20_newsgroups(random_state)
    elif dataset == "20NewsGroups-raw":
        X_train, X_test, y_train, y_test, labels = get_20_newsgroups_bert(random_state)
    elif dataset == "Covertype":
        X_train, X_test, y_train, y_test, labels = get_covtype(random_state)
    elif dataset == "Iris":
        X_train, X_test, y_train, y_test, labels = get_iris(random_state)
    else:
        raise ValueError(f"Dataset not recognized: {dataset}")
    
    return X_train, X_test, y_train, y_test, labels

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, labels = get_dataset("20NewsGroups-bert", 0)
    print()