"""Extract features from textual documents using gensim utilities.
"""

import multiprocessing
from pathlib import Path
from pprint import pprint  # pylint: disable=unused-import
import sys  # pylint: disable=unused-import
from tempfile import NamedTemporaryFile
from typing import Generator, Iterable, List, Tuple

from gensim import downloader
from gensim.models.doc2vec import Doc2Vec, TaggedLineDocument
from gensim.models.fasttext import FastText
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import LineSentence, Word2Vec
from gensim.utils import tokenize
import numpy as np

from active_learning.feature_extractors.base import FeatureExtractor


# Some documents contain no tokens after preprocessing and tokenization
# We replace these tokenized documents with a single token, which promotes
# usage of feature extraction techniques
empty_token = "<EMPTY>"


class GensimFeatureExtractor(FeatureExtractor):
    """Feature extraction methods for architectures provided by gensim.

    Attributes
    ----------
    model : Union[Doc2Vec, Word2Vec, FastText, KeyedVectors]
        The gensim model to perform word inference upon. If a pretrained model is used,
            will be a KeyedVectors object. If a training a model from scratch, will be
            one of the three gensim models.
    """

    def __init__(self, model: str, **kwargs):
        """Instantiate the feature extractor.

        Parameters
        ----------
        model : str
            Base model architecture to use. One of 'Doc2Vec', 'Word2Vec', or 'FastText' for
                training base models or any of the pretrained models from the
                gensim data repository
        **kwargs
            Keyword arguments passed to the constructor of the gensim base model

        Raises
        ------
        ValueError
            If the model is not recongized.
        """
        workers = multiprocessing.cpu_count() - 1 or 1

        valid_models = {"Doc2Vec", "Word2Vec", "FastText"}.union(downloader.info()["models"].keys())
        if model == "Doc2Vec":
            self.model = Doc2Vec(workers=workers, **kwargs)
        elif model == "Word2Vec":
            self.model = Word2Vec(workers=workers, **kwargs)
        elif model == "FastText":
            self.model = FastText(workers=workers, **kwargs)
        elif model in set(downloader.info()["models"].keys()):
            self.model = downloader.load("glove-twitter-25")
        else:
            raise ValueError(
                f"Unknown feature representation/model type: {model}. "
                f"Valid options are: {valid_models}."
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

        # If the model is a KeyedVectors object, its pretrained and training is not needed
        if not isinstance(self.model, KeyedVectors):
            X_train = self.train(X_train)

        # Vectorize the data using the learned model
        X_train = self.vectorize(X_train)
        X_test = self.vectorize(X_test)

        # Return the vectorized data
        return X_train, X_test

    def train(self, X_train: Iterable[List[str]]) -> Generator[List[str], None, None]:
        """Train the model if not using a pretrained model.

        Parameters
        ----------
        X_train : Iterable[List[str]]
            Tokenized and processed text data

        Returns
        -------
        Generator[List[str], None, None]
            The same tokenized and processed text data that was passed in as an argument
        """

        try:
            # Save data to temporary file to support multi-pass streaming
            f_train = NamedTemporaryFile(delete=False)
            with open(f_train.name, "w", encoding="utf8") as f:
                f.writelines((" ".join(x) + "\n" for x in X_train))

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

            # Re-acquire the the training and test data from the files to return new generator
            X_train = (line.split() for line in open(f_train.name, "r", encoding="utf8"))

        except Exception as e:
            raise e

        # Delete the temporary files
        finally:
            p_train = Path(f_train.name)
            if p_train.exists():
                p_train.unlink()

        return X_train

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

        if isinstance(self.model, (Word2Vec, FastText)):
            wv = self.model.wv
        elif isinstance(self.model, KeyedVectors):
            wv = self.model
        else:
            raise TypeError(
                f"Unexpected type for self.model: {type(self.model)}. "
                f"Expected types are Doc2Vec, Word2Vec, FastText, or KeyedVectors."
            )

        X_ = []
        for x in X:
            embeddings = [wv[w] for w in x if w in wv]
            if embeddings:
                X_.append(np.mean(embeddings, axis=0))
            else:
                X_.append(np.zeros(wv.vector_size))

        return np.array(X_)
