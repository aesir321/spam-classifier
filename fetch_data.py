# Taken from https://github.com/ageron/handson-ml2/blob/master/03_classification.ipynb
import os
import urllib.request
import tarfile

DOWNLOAD_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"
HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"
SPAM_PATH = os.path.join("data", "spam")


def fetch_spam_data(ham_url=HAM_URL, spam_url=SPAM_URL, spam_path=SPAM_PATH):
    if not os.path.isdir(spam_path):
        os.makedirs(spam_path)

    for filename, url in (("ham.tar.bz2", ham_url), ("spam.tar.bz2", spam_url)):
        path = os.path.join(spam_path, filename)

        if not os.path.isfile(path):
            urllib.request.urlretrieve(url, path)

        tar_bz2_file = tarfile.open(path)
        tar_bz2_file.extractall(path=spam_path)
        tar_bz2_file.close()


def delete_unwanted_files():
    os.remove(os.path.join(SPAM_PATH, "easy_ham/cmds"))
    os.remove(os.path.join(SPAM_PATH, "spam/cmds"))


if __name__ == "__main__":
    fetch_spam_data()
    delete_unwanted_files()
