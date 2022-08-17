import os
import statistics
import joblib

from settings import TRAIN_FINAL, TRAIN_ONCE, SAVE_TO_DISK, STORAGE_DIR, CV_SPLITS, CUSTOM_NAME_SUFFIX

from utils.get_documents import create_training_dataframe
from utils.preprocessing import text_preprocessing

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

from nltk.corpus import stopwords

# RUN ONCE:
# import nltk
# nltk.download("stopwords")

class Train_Votingclassifier():
    def __init__(self):
        df = create_training_dataframe(use_saved=False, clf="votingclassifier")

        # Include German word boundaries instead of the standard (?u)\b\w\w+\b (which doesn't allow german mutated vowels)
        countVect = CountVectorizer(min_df=0.1, max_df=0.5, stop_words=stopwords.words('german'), ngram_range=(1, 2), token_pattern=r"(?u)(?<![äöüÄÖÜß\w])([äöüÄÖÜß\w]+)(?![äöüÄÖÜß\w])")

        transformer = FunctionTransformer(text_preprocessing)


        text_clf = Pipeline(steps=[
            ("trans", transformer),
            ('vect', countVect),
            ('tfidf', TfidfTransformer()),
            # Parameter optimized with GridSearchCV, see below
            ('clf', SVC(C=1.0, kernel='linear', degree=2, gamma='auto', probability=True))
        ])

        """
        GridSearchCV Hyperparameter Optimization
        
        params = {
            'clf__C': [0.1, 0.5, 1.0, 10, 100.0],
            'clf__degree': [2, 3, 4]
        }
        grid = GridSearchCV(estimator=text_clf, param_grid=params, cv=10, scoring="accuracy")
        
        grid.fit(X=df["content"], y=df["category"])
        >>> clf__C = 1, clf__degree = 2
        """

        text_clf2 = Pipeline(steps=[
            ("trans", transformer),
            ('vect', countVect),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB()),
        ])

        text_clf3 = Pipeline(steps=[
            ("trans", transformer),
            ('vect', countVect),
            ('tfidf', TfidfTransformer()),
            ('clf', RandomForestClassifier())
        ])

        voting_clf = Pipeline(steps=[('clf4', VotingClassifier(estimators=[("svc", text_clf), ("mnb", text_clf2), ("rfc", text_clf3)], weights=[1, 1, 1], voting="soft"))])


        if TRAIN_FINAL:
            voting_clf.fit(X=df["content"], y=df["category"])

            if SAVE_TO_DISK:
                joblib.dump(voting_clf, os.path.join(STORAGE_DIR, f'voting_classifier{CUSTOM_NAME_SUFFIX}.pkl'))

            self.accuracies = []


        elif not TRAIN_ONCE and not SAVE_TO_DISK:
            # Using the short form of the StratifiedKFold
            accuracies = cross_val_score(voting_clf, df["content"], df["category"], scoring='accuracy', cv=CV_SPLITS)
            print(statistics.mean(accuracies))
            print(accuracies)
            self.accuracies = accuracies

        else:  # NOT TRAIN_FINAL AND NOT TRAIN_ONCE and SAVE_TO_DISK:
            skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True)
            split_index = 1
            accuracies = []
            for train_index, test_index in skf.split(df["content"], df["category"]):
                # no need for transformation like in doc2vec since it's being transformed through the pipeline
                X_train, X_test = df["content"][train_index], df["content"][test_index]
                y_train, y_test = df["category"][train_index], df["category"][test_index]

                voting_clf.fit(X=X_train, y=y_train)
                accuracy = voting_clf.score(X=X_test, y=y_test)
                accuracies.append(accuracy)

                if split_index == CV_SPLITS:  # save the last split
                    joblib.dump(voting_clf, os.path.join(STORAGE_DIR, f'voting_classifier{CUSTOM_NAME_SUFFIX}.pkl'))
                else:
                    split_index += 1

            print(statistics.mean(accuracies))
            print(accuracies)
            self.accuracies = accuracies

    def get_training_accuracies(self):
        return self.accuracies
