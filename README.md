# Spam Classifier

A spam classifer based on the [spamassassin dataset](https://spamassassin.apache.org/old/publiccorpus/).

## Requirements

* Built with Python 3.9.5

Some problems were experienced with installing the pip package wordcloud on Windows.  This was solved following the instructions [here](https://shuchanel.com/resolve-python-pip-error-microsoft-visual-c-14-0-or-greater-is-required/).

## Datasets

Data description taken from spamassassin README

  - spam: 500 spam messages, all received from non-spam-trap sources.

  - easy_ham: 2500 non-spam messages.  These are typically quite easy to
    differentiate from spam, since they frequently do not contain any spammish
    signatures (like HTML etc).

  - hard_ham: 250 non-spam messages which are closer in many respects to
    typical spam: use of HTML, unusual HTML markup, coloured text,
    "spammish-sounding" phrases etc.

  - easy_ham_2: 1400 non-spam messages.  A more recent addition to the set.

  - spam_2: 1397 spam messages.  Again, more recent.

Total count: 6047 messages, with about a 31% spam ratio.

## Usage instructions

## Acknowledgements

Parts of this repository are based upon work from the book Hands-On Machine Learning with Scikit-Learn, Keras and Tensorflow by Aurelien Geron.
