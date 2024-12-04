from collections import Counter
import re

import numpy as np


class TFIDF:
    """
    Ref source: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer
    """

    def __init__(self):
        self.word_counter = None
        self.index_to_word = None
        self.word_to_index = None
        self.document_frequency = None
        self.n_docs = 0
        self.n_vocab = 0

    def _clean_data(self, data: list[str]) -> list[str]:
        """Cleans and preprocesses data by removing non-alphabet characters and lowering case."""
        return [
            " ".join(re.findall(r"[a-zA-Z]+", sentence)).lower() for sentence in data
        ]

    def _build_vocab(self, cleaned_data: list[str]) -> list[str]:
        """Builds vocabulary and mappings from cleaned data."""
        word_list = [word for sentence in cleaned_data for word in sentence.split()]
        self.word_counter = dict(sorted(Counter(word_list).items()))
        self.index_to_word = {
            idx: word for idx, word in enumerate(self.word_counter.keys())
        }
        self.word_to_index = {
            word: idx for idx, word in enumerate(self.word_counter.keys())
        }
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
        transformed_matrix = np.array(
            [self._calculate_tfidf(sentence) for sentence in cleaned_data]
        )
        return transformed_matrix

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


class MultinomialNB:
    """
    Ref source: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB
    """

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
            self.feature_likelihoods[cls] = (np.sum(X_cls, axis=0) + 1) / (
                np.sum(X_cls) + X.shape[1]
            )

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


class LabelBinarizer:
    """
    Ref source: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html
    """

    def __init__(self):
        self.rank_mapping = None

    def fit(self, y):
        unique_label = np.unique(y)
        self.rank_mapping = {label: rank for rank, label in enumerate(unique_label)}
        return self

    def transform(self, y):
        if len(self.rank_mapping) == 2:
            result = np.zeros((len(y), 1))
            for i in range(len(y)):
                if y[i] == list(self.rank_mapping.keys())[0]:
                    result[i] = 0
                else:
                    result[i] = 1
            return result

        result = np.zeros((len(y), len(self.rank_mapping)))
        for i in range(len(y)):
            if self.rank_mapping.get(y[i], None) is not None:
                col = self.rank_mapping.get(y[i])
                result[i][col] = 1
        return result

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)


def chi_square(X: np.ndarray, y: np.ndarray):
    """
    Ref source: https://github.com/ethen8181/machine-learning/blob/master/text_classification/chisquare.ipynb
    """
    y = LabelBinarizer().fit_transform(y)

    observed = np.dot(y.T, X)

    class_prob = np.mean(y, axis=0).reshape(1, -1)
    feature_count = np.sum(X, axis=0).reshape(1, -1)
    expected = np.dot(class_prob.T, feature_count)

    chi_square = np.sum((observed - expected) ** 2 / expected, axis=0)
    return chi_square


class SMOTE:
    """
    Ref sources: https://www.jair.org/index.php/jair/article/view/10302
    """

    def __init__(self, k_neighbors: int = 5, random_state: int = 42):
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.n_attrs = None
        self.synthetic = None
        self.new_index = None
        np.random.seed(self.random_state)

    def _calculate_neighbors(self, data_point, X) -> np.ndarray:
        distance = np.linalg.norm(X - data_point, axis=1)
        sorted_indices = np.argsort(distance)
        return sorted_indices[1 : self.k_neighbors]

    def _populate(self, N, i, nn_array):
        while N:
            nn = np.random.randint(len(nn_array))
            for attr in range(self.n_attrs):
                diff = self.X_minority[nn_array[nn]][attr] - self.X_minority[i][attr]
                gap = np.random.uniform(0, 1)
                self.synthetic[self.new_index][attr] = self.X_minority[i][attr] + (
                    gap * diff
                )
            self.new_index += 1
            N -= 1

    def fit_resample(
        self, X: np.ndarray, y: np.ndarray, sampling_class: int, N: int = 100
    ) -> tuple[np.ndarray, np.ndarray]:
        self.n_attrs = X.shape[1]
        self.X_minority = X[y == sampling_class]

        T = len(self.X_minority)
        if N < 100:
            T = int((N / 100) * T)
            N = 100
        N = int(N / 100)

        self.synthetic = np.zeros((T * N, self.n_attrs))
        self.new_index = 0

        for i in range(T):
            data_point = self.X_minority[i]
            nn_array = self._calculate_neighbors(data_point, self.X_minority)
            self._populate(N, i, nn_array)

        X_resampled = np.vstack((X, self.synthetic))
        y_synthethic = np.ones(self.new_index) * sampling_class
        y_resampled = np.concatenate((y, y_synthethic))

        return X_resampled, y_resampled


def split_data(X: np.ndarray, y: np.ndarray, test_size=0.2, random_state=None):
    """
    Split data into train, test
    """
    if random_state:
        np.random.seed(random_state)

    indices = np.random.permutation(X.shape[0])
    test_size = int(X.shape[0] * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy score.
    Formula: (TP + TN) / (TP + FP + TN + FN)
    """
    return float(np.mean(y_true == y_pred))
