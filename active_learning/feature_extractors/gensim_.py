"""Extract features from complex data objects, such as text documents.

TODO: set the random seed for the feature extractors.
"""

import multiprocessing
from pathlib import Path
from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from tempfile import NamedTemporaryFile
from typing import Iterable, List, Tuple, Union

from gensim.models.fasttext import FastText
from gensim.models.doc2vec import Doc2Vec, TaggedLineDocument
from gensim.models.word2vec import LineSentence, Word2Vec
from gensim.utils import tokenize
import numpy as np

from active_learning.feature_extractors.base import FeatureExtractor


# Some documents contain no tokens after preprocessing and tokenization
# We replace these tokenized documents with a single token, which promotes
# usage of feature extraction techniques
empty_token = "<EMPTY>"


class GensimFeatureExtractor(FeatureExtractor):
    """Feature extraction methods for architectures provided by gensim."""

    def __init__(self, feature_rep: str, **kwargs):
        """Instantiate the feature extractor.

        Parameters
        ----------
        feature_rep : str
            Base model architecture to use. One of 'Doc2Vec', 'Word2Vec', or 'FastText'
        **kwargs
            Keyword arguments passed to the constructor of the gensim base model

        Raises
        ------
        ValueError
            If the feature_rep is not recongized.
        """
        workers = multiprocessing.cpu_count() - 1 or 1

        if feature_rep == "Doc2Vec":
            self.model = Doc2Vec(workers=workers, **kwargs)
        elif feature_rep == "Word2Vec":
            self.model = Word2Vec(workers=workers, **kwargs)
        elif feature_rep == "FastText":
            self.model = FastText(workers=workers, **kwargs)
        else:
            raise ValueError(
                f"Unknown feature representation: {feature_rep}. "
                f"Valid options are: 'Doc2Vec', 'Word2Vec', or 'FastText'."
            )

    def extract_features(
        self, X_train: Iterable[str], X_test: Iterable[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract the features from a text dataset.

        Note that this process was made more complex (temp file I/O) to support streaming.

        Parameters
        ----------
        X_train : Iterable[str]
            A one dimensional iterable of textual training data.
        X_test : Iterable[str]
            A one dimensional iterable of textual test data.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Two dimensional feature representations of the input corpora.
        """
        # Tokenize the input data
        X_train = (list(tokenize(x, lowercase=True)) for x in X_train)
        X_test = (list(tokenize(x, lowercase=True)) for x in X_test)

        # Apply the empty token to empty documents
        X_train = (x if x else [empty_token] for x in X_train)
        X_test = (x if x else [empty_token] for x in X_test)

        # Create temporary files to support multi-pass streaming
        f_train = NamedTemporaryFile(delete=False)
        f_test = NamedTemporaryFile(delete=False)

        # Try block cleans up temporary files in case of exception
        try:
            # Save data to temporary files
            with open(f_train.name, "w", encoding="utf8") as f:
                f.writelines((" ".join(x) + "\n" for x in X_train))
            with open(f_test.name, "w", encoding="utf8") as f:
                f.writelines((" ".join(x) + "\n" for x in X_test))

            # Prep data into gensim streaming objects (conceptually different from X_train)
            if isinstance(self.model, Doc2Vec):
                corpus = TaggedLineDocument(f_train.name)
            else:
                corpus = LineSentence(f_train.name)

            # Build the vocab and train the model on the corpus
            self.model.build_vocab(corpus)
            self.model.train(
                corpus, total_examples=self.model.corpus_count, epochs=self.model.epochs
            )

            # Re-acquire the the training and test data from the files
            X_train = (line.split() for line in open(f_train.name, "r", encoding="utf8"))
            X_test = (line.split() for line in open(f_test.name, "r", encoding="utf8"))

            # Vectorize the data using the learned model
            X_train = self.vectorize(X_train)
            X_test = self.vectorize(X_test)

        # Delete the temporary files before raising any potention exceptions
        except Exception as e:
            p_train, p_test = Path(f_train.name), Path(f_test.name)
            if p_train.exists():
                p_train.unlink()
            if p_test.exists():
                p_test.unlink()
            raise e

        # Return the vectorized data
        return X_train, X_test

    def vectorize(self, X: Iterable[List[str]]) -> np.ndarray:
        """Convert the input corpus into a vector representation by performing model inference.

        Parameters
        ----------
        X : Iterable[List[str]]
            Tokenized and processed text data

        Returns
        -------
        np.ndarray
            A two-dimensional numerical representation of the input corpus
        """
        if isinstance(self.model, Doc2Vec):
            return np.array([self.model.infer_vector(x) for x in X])

        X_ = []
        for x in X:
            embeddings = [self.model.wv[w] for w in x if w in self.model.wv]
            if embeddings:
                X_.append(np.mean(embeddings, axis=0))
            else:
                X_.append(np.zeros(self.model.vector_size))

        return np.array(X_)
