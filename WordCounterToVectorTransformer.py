from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter


class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size

    def fit(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                # Not sure why this is min(count, 10) and not just 10
                total_count[word] += count  # min(count, 10)
        most_common = total_count.most_common()[: self.vocabulary_size]

        # _ to indicate learned parameter
        self.vocabulary_ = {
            word: index + 1 for index, (word, _) in enumerate(most_common)
        }
        return self

    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix(
            (data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1)
        )
