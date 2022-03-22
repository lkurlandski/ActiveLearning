#file to convert text data to vectors
from gensim.models import Word2Vec
import numpy as np
from datasets import load_dataset

def w2v_vectorize(X_test, X_train):
    #create the vectors using boff

    print(X_test)
    X_test = [sentence.split() for sentence in X_test]
    X_train = [sentence.split() for sentence in X_train]
    X_combined = X_test + X_train
    print(X_test)

    #X_combined_numpy = np.asarray(X_combined)
    #print(X_combined_numpy.shape[0])
    #print(X_test)
    
    vectors = Word2Vec(sentences=X_combined, min_count=1, window=15, epochs=50)
    #print(vectors.wv['integrity'])

    X_test_vectors = []
    X_train_vectors = []
    X_test_vectors_iter = []
    X_train_vectors_iter = []
    #create a list of lists of W2V vectors using X_test and X_train
    for sentence in X_test:
        for token in sentence[:300]:
            try:
                X_test_vectors_iter.append(vectors.wv[token])
                #look at the function in sequencer, add zeros and flatten
            except:
                print("Could not find token", token)
                pass

        last_pieces = 300 - len(X_test_vectors_iter)
        for i in range(last_pieces):
            X_test_vectors_iter.append(np.zeros(100,))
        X_test_vectors.append(np.asarray(X_test_vectors_iter).flatten())
        X_test_vectors_iter.clear()

    for sentence in X_train:
        for token in sentence[:300]:
            try:
                X_train_vectors_iter.append(vectors.wv[token])
                #look at the function in sequencer, add zeros and flatten
            except:
                print("Could not find token", token)
                pass

        last_pieces = 300 - len(X_train_vectors_iter)
        for i in range(last_pieces):
            X_train_vectors_iter.append(np.zeros(100,))
            
        X_train_vectors.append(np.asarray(X_train_vectors_iter).flatten())
        X_train_vectors_iter.clear()

    #print(np.asarray(X_test_vectors).shape)
    #print(np.asarray(X_train_vectors).shape)

    return np.asarray(X_test_vectors), np.asarray(X_train_vectors)


#test_dataset = load_dataset('emotion', split = 'test')
#labels = test_dataset.features['label'].names
#X_test = test_dataset['text']
#y_test = test_dataset['label']

#test_dataset = load_dataset('emotion', split = 'train')
#X_train = test_dataset['text']
#y_train = test_dataset['label']
#w2v_vectorize(X_test, X_train)
    