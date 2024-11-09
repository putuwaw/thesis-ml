# %%
from collections import Counter
import re
import numpy as np

# for testing
from sklearn import naive_bayes, feature_extraction

# %%
corpus = ['this is the first document',
          'this document is the second document',
          'and this is the third one',
          'is this the first document']

# %%
class TFIDF:
    def __init__(self):
        self.word_counter = None
        self.index_to_word = None
        self.word_to_index = None
        self.document_frequency = None
        self.n_docs = 0
        self.n_vocab = 0

    def _clean_data(self, data: list[str]) -> list[str]:
        """Cleans and preprocesses data by removing non-alphabet characters and lowering case."""
        return [' '.join(re.findall(r'[a-zA-Z]+', sentence)).lower() for sentence in data]

    def _build_vocab(self, cleaned_data: list[str]) -> list[str]:
        """Builds vocabulary and mappings from cleaned data."""
        word_list = [word for sentence in cleaned_data for word in sentence.split()]
        self.word_counter = dict(sorted(Counter(word_list).items()))
        self.index_to_word = {idx: word for idx, word in enumerate(self.word_counter.keys())}
        self.word_to_index = {word: idx for idx, word in enumerate(self.word_counter.keys())}
        self.n_vocab = len(self.word_counter)

    def _calculate_document_frequency(self, cleaned_data: list[str]):
        """Calculates document frequency for each word."""
        self.document_frequency = {word: 0 for word in self.word_to_index.keys()}
        for sentence in cleaned_data:
          splitted_sentence = sentence.split(" ")
          unique_word = np.unique(splitted_sentence)
          for word in unique_word:
            self.document_frequency[word] += 1

    def _calculate_tfidf(self, sentence: str):
        """Calculates the TF-IDF vector for a single sentence."""
        counter = Counter(sentence.split())
        row_sum = 0
        tfidf_vector = np.zeros(self.n_vocab)

        for j in range(self.n_vocab):
            word = self.index_to_word[j]
            # term frequency only caculated from freq on the sentence
            # same as BoW (sklearn references)
            tf = counter.get(word, 0)
            df = self.document_frequency.get(word, 0)
            # references: https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
            # adding 1 on n and df -> smoothing
            idf = np.log((1 + self.n_docs) / (1 + df)) + 1
            tf_idf = tf * idf

            row_sum += np.square(tf_idf)
            tfidf_vector[j] = tf_idf

        l2_norm = np.sqrt(row_sum)
        if l2_norm > 0:
            tfidf_vector /= l2_norm

        return tfidf_vector

    def fit(self, data):
        """Fits the model on the data"""
        cleaned_data = self._clean_data(data)
        self.n_docs = len(cleaned_data)
        self._build_vocab(cleaned_data)
        self._calculate_document_frequency(cleaned_data)
        return self

    def transform(self, data):
        """Transforms new data based on the already fitted vocabulary and document frequencies."""
        if self.word_to_index is None or self.document_frequency is None:
            raise ValueError("The TFIDF model must be fitted before calling transform.")

        cleaned_data = self._clean_data(data)
        transformed_matrix = np.array([self._calculate_tfidf(sentence) for sentence in cleaned_data])
        return transformed_matrix

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

# %%
tfidf = TFIDF()
X_tfidf = tfidf.fit_transform(corpus)
X_tfidf

# %%
tfidf_sklearn = feature_extraction.text.TfidfVectorizer()
X_sklearn = tfidf_sklearn.fit_transform(corpus).toarray()
X_sklearn

# %%
assert np.array_equal(X_tfidf, X_sklearn)

# %%
class MultinomialNB:
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes = np.unique(y)
        self.class_priors = {}
        self.feature_likelihoods = {}
        self.join_log_likelihoods = []

        # for each class calculate prior and likelihoods
        for cls in self.classes:
            X_cls = X[y == cls]

            # class prior = total this class / total dataset
            self.class_priors[cls] = X_cls.shape[0] / X.shape[0]

            # feature likelihoods = array with cols eq to total feature
            # for each feature, sum all occurrence of that feature
            # divide it by (sum features in that class + total feature)
            self.feature_likelihoods[cls] = (np.sum(X_cls, axis=0) + 1) / (np.sum(X_cls) + X.shape[1])

    def predict(self, X):
        predictions = []

        # for each data
        for x in X:
            posteriors = {}

            # for each class
            for cls in self.classes:
                # normal
                # posterior = (self.class_priors[cls])
                # posterior *= np.prod(np.power(self.feature_likelihoods[cls], x))

                # problem with normal is too small result, because fraction * fraction
                # for example 0.3 * 0.5 * ......
                # log version
                log_posterior = np.log(self.class_priors[cls])

                # math time
                # log(A^B) = log(A) * B
                # log(likelihoods ^ x ) = log(likelihoods) * x
                # log(a*b) = log(a) + log(b), then:
                # np.prod => np.sum
                log_posterior += np.sum(np.log(self.feature_likelihoods[cls]) * x)

                # add to class posterior
                posteriors[cls] = np.float64(log_posterior)

            # predict the class by choosing the highst log posterior
            self.join_log_likelihoods.append(list(posteriors.values()))
            predictions.append(max(posteriors, key=posteriors.get))
            # same as np.argmax(list(posteriors.values()))

        return np.array(predictions)

# %%
rng = np.random.RandomState(1)
X = rng.randint(5, size=(6, 100))
y = np.array([1, 2, 3, 4, 5, 6])

# %%
mnb = MultinomialNB()

# %%
mnb.fit(X, y)
print(mnb.predict(X[2:3]))

# %%
clf = naive_bayes.MultinomialNB()
clf.fit(X, y)
pred = (clf.predict_joint_log_proba(X[2:3]))
pred

# %%
jll = np.array(mnb.join_log_likelihoods)
jll

# %%
assert np.allclose(jll, pred)


