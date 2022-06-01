#!/home/hpc/schmij12/fineTune/ActiveLearning/env/bin/python -u

# SBATCH --chdir=/home/hpc/schmij12/fineTune/ActiveLearning
# SBATCH --output=/home/hpc/schmij12/fineTune/ActiveLearning/slurm/jobs/job.name.%A.out
# SBATCH --constraint=skylake|broadwell
# SBATCH --job-name=finetune
# SBATCH --partition=gpu
# SBATCH --cpus-per-task=1
# SBATCH --mem=32G

# sbatch script up top
# change directories to your own to run as sbatch script

# This file is to try and implement a BERT fine-tuning custom estimator (scikit-learn) that can be used with modAL. This would be used as the estimator for the learner.
# An estimator must implement .fit, .predict_proba, and .predict methods.
# More documentation on scikit-learn custom estimators at https://scikit-learn.org/stable/developers/develop.html
# This currently has .fit working with the yelp dataset from huggingface
# I followed guides for bert fine-tuning from huggingface and implemented that in .fit
# predict and predict_proba need work
# https://github.com/charles9n/bert-sklearn is an example of someone else's implementation
# usually predict and predict_proba for fine-tuning implement the softmax function
# needs to be tested more and with more datasets, to eventually be used in the codebase as an option for the estimator


# import dependencies
import numpy as np
import sys
import torch
from active_learning import feature_extractors, dataset_fetchers

import torch.nn.functional as F
from tqdm import tqdm

from sklearn.base import BaseEstimator

from transformers import BertTokenizerFast, get_scheduler, BertForSequenceClassification
from torch.optim import AdamW
from datasets import load_dataset

from torch.utils.data import DataLoader


pbar = tqdm


class sturcc_to_holder:
    def __init__(self, list_files):
        self.list_of_files = list_files

    # with tensors
    def __getitem__(self, idx):
        aa = torch.load(self.list_of_files[idx])
        return aa[0], aa[1], aa[2]

    def __len__(self):
        return len(self.list_of_files)

    # without toensors
    def __getitemnumpy__(self, idx):
        aa = np.load(self.list_of_files[idx], allow_pickle=True)
        cc = aa.data.obj.tolist()
        c1 = cc.input_ids
        c2 = cc.attention_mask
        c3 = cc.label
        return torch.tensor(c1), torch.tensor(c2), c3


def unpack_text_pairs(X):
    """
    Unpack text pairs
    """
    if X.ndim == 1:
        texts_a = X
        texts_b = None
    else:
        texts_a = X[:, 0]
        texts_b = X[:, 1]

    return texts_a, texts_b


def to_numpy(X):
    """
    Convert input to numpy ndarray
    """
    if hasattr(X, "iloc"):  # pandas
        return X.values
    elif isinstance(X, list):  # list
        return np.array(X)
    elif isinstance(X, np.ndarray):  # ndarray
        return X
    else:
        raise ValueError("Unable to handle input type %s" % str(type(X)))


def unpack_data(X, y=None):
    """
    Prepare data
    """
    X = to_numpy(X)
    texts_a, texts_b = unpack_text_pairs(X)

    if y is not None:
        y = to_numpy(y)
        labels = y
        return texts_a, texts_b, labels
    else:
        return texts_a, texts_b


def tokenize_function(examples):

    return bert_tokenizer(examples["text"], padding="max_length", truncation=True)


# the actual bert fine tuning custom estimator
class BertClassifer(BaseEstimator):

    # constructor takes in a bert model and bert tokenizer
    def __init__(self, bert_model, bert_tokenizer):
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer

    # takes in a dataset composed of features and labels currently, should be changed to X and Y (maybe as an option
    # ) most likely or a differnt way to handle
    def fit(self, X):

        # Check that X and y have correct shape (see scikit-learn documentation)
        # X, y = check_X_y(X, y)

        train_dataloader = DataLoader(X, shuffle=True, batch_size=8)

        optimizer = AdamW(bert_model.parameters(), lr=5e-5)

        num_epochs = 3

        num_training_steps = num_epochs * len(train_dataloader)

        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # for debugging, ideally can get it to work with the gpu
        print(device)

        bert_model.to(device)

        from tqdm.auto import tqdm

        progress_bar = tqdm(range(num_training_steps))

        bert_model.train()

        for epoch in range(num_epochs):
            for batch in train_dataloader:

                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = bert_model(**batch)

                loss = outputs.loss

                loss.backward()

                optimizer.step()

                lr_scheduler.step()

                optimizer.zero_grad()

                progress_bar.update(1)

        # Return the classifier
        return self

    def predict(self, X) -> np.ndarray:  # needs to be implelemented correctly

        y_pred = np.argmax(self.predict_proba(X), axis=1)
        y_pred = np.array([self.id2label[y] for y in y_pred])
        return y_pred

    def predict_proba(self, X) -> np.ndarray:  # needs to be implemented correctly

        """probas = self.predict_proba(X)
        if not all(p.shape == (X.shape[0], 2) for p in probas):
            raise ValueError(
                "Expected every probability to have shape (n_samples, 2) not "
                f"{(X.shape[0], 2)}"
            )"""
        texts_a, texts_b = unpack_data(X)

        list_of_files = []
        your_data = sturcc_to_holder(list_files=list_of_files)

        dss = DataLoader([texts_a, texts_b], batch_size=16, shuffle=True)
        device = torch.device("cpu")

        probs = []
        sys.stdout.flush()
        batch_iter = pbar(dss, desc="Predicting", leave=True)

        for batch in batch_iter:
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                logits = self.model(*batch)
                prob = F.softmax(logits, dim=-1)
            prob = prob.detach().cpu().numpy()
            probs.append(prob)
        sys.stdout.flush()
        return np.vstack(tuple(probs))


# initalize model and tokenizer from pretrained huggingface
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)
bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# initalize model
model = BertClassifer(bert_model, bert_tokenizer)

# load dataset from huggingface
dataset = load_dataset("yelp_review_full")

# get dataset in correct format, based on hugginface fine tuning documentation. Should be expanded to work with any type of dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))


# Where i was testing the methods of the estimator

model.fit(small_train_dataset)

model.predict_proba(small_eval_dataset)

# eventually test inside active learning codebase, particulary with the learn method
# learn(model, uncertainty_batch_sampling, batch_size=batch_size, unlabeled_pool=Pool(x_train, y_train), test_set = Pool(x_test, y_test))
