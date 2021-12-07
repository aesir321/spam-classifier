from sklearn.base import BaseEstimator, TransformerMixin
from bs4 import BeautifulSoup
import urlextract
import re
from collections import Counter
import numpy as np
import nltk


class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        strip_headers=True,
        lower_case=True,
        remove_punctuation=True,
        replace_urls=True,
        replace_numbers=True,
        stemming=True,
    ):
        self.strip_headers = strip_headers
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming
        self.url_extractor = urlextract.URLExtract()
        self.stemmer = nltk.PorterStemmer()

    def __email_to_text(self, email):
        html = None
        for part in email.walk():
            ctype = part.get_content_type()
            # TODO: Can we make this better with get_body()?
            if not ctype in ("text/plain", "text/html"):
                continue
            try:
                content = part.get_content()
            except:  # in case of encoding issues
                content = str(part.get_payload())
            if ctype == "text/plain":
                return content
            else:
                html = content
        if html:
            return BeautifulSoup(html, "html.parser").get_text()

    def __replace_urls(self, text):
        urls = list(set(self.url_extractor.find_urls(text)))
        urls.sort(key=lambda url: len(url), reverse=True)
        for url in urls:
            text = text.replace(url, " URL ")
        return text

    def __replace_numbers(self, text):
        return re.sub(r"\d+(?:\.\d*)?(?:[eE][+-]?\d+)?", "NUMBER", text)

    def __remove_punctuation(self, text):
        return re.sub(r"\W+", " ", text, flags=re.M)

    def __stem(self, word_counts):
        stemmed_word_counts = Counter()
        for word, count in word_counts.items():
            stemmed_word = self.stemmer.stem(word)
            stemmed_word_counts[stemmed_word] += count
        return stemmed_word_counts

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = []
        for email in X:
            # if there is a problem we just make it empty string
            text = self.__email_to_text(email) or ""
            if self.lower_case:
                text = text.lower()
            # Why the check for none on urlextractor?
            if self.replace_urls and self.url_extractor is not None:
                text = self.__replace_urls(text)
            if self.replace_numbers:
                text = self.__replace_numbers(text)
            if self.remove_punctuation:
                text = self.__remove_punctuation(text)
            word_counts = Counter(text.split())
            if self.stemming and self.stemmer is not None:
                word_counts = self.__stem(word_counts)
            X_transformed.append(word_counts)
        return np.array(X_transformed)
