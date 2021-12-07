from collections import Counter
import os
import email
import email.policy

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import urlextract


DOWNLOAD_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"
HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"
SPAM_PATH = os.path.join("data", "spam")
HAM_DIR = os.path.join(SPAM_PATH, "easy_ham")
SPAM_DIR = os.path.join(SPAM_PATH, "spam")
RANDOM_STATE = 42


def load_email(is_spam, filename, spam_path=SPAM_PATH):
    directory = "spam" if is_spam else "easy_ham"

    with open(os.path.join(spam_path, directory, filename), "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)


def create_email_list_from_dir(dir, is_spam=False):
    filenames = [name for name in os.listdir(dir)]
    emails = [load_email(is_spam=is_spam, filename=name) for name in filenames]

    return emails


ham_emails = create_email_list_from_dir(HAM_DIR)
spam_emails = create_email_list_from_dir(SPAM_DIR, is_spam=True)

X = np.array(ham_emails + spam_emails, dtype=object)
y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

count_vect = CountVectorizer()
