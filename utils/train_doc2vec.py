import os
import numpy as np
import joblib

from settings import TRAIN_FINAL, CV_SPLITS, EPOCHS, TRAIN_ONCE, SAVE_TO_DISK, STORAGE_DIR

from utils.get_documents import create_training_dataframe
from utils.preprocessing import text_preprocessing
from utils.classification_report import heatconmat

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import classification_report, accuracy_score
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


# see source[2] (only parts were adopted)
def train_model(model : Doc2Vec, tagged_tr : list[TaggedDocument], y_train, save_to_disk=True):
    model.build_vocab(tagged_tr)
    for epoch in range(EPOCHS):
        print(f'Training Epoch {epoch + 1}')
        model.train(tagged_tr,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)

    X_train = np.array([model.dv[str(i)] for i in range(len(tagged_tr))])

    lrc = LogisticRegression(C=5, multi_class='multinomial', solver='saga', max_iter=1500)
    lrc.fit(X_train, y_train)

    if save_to_disk:
        model.save(os.path.join(STORAGE_DIR, 'doc2vec.model'))
        joblib.dump(lrc, os.path.join(STORAGE_DIR, 'lrc.pkl'))
    return model, lrc


# see source[2]
def test_model(model : Doc2Vec, lrc : LogisticRegression, tagged_test : list[TaggedDocument], y_test):
    X_test = np.array([model.infer_vector(tagged_test[i][0]) for i in range(len(tagged_test))])
    y_pred = lrc.predict(X_test)

    print(classification_report(y_true=y_test, y_pred=y_pred))
    heatconmat(y_true=y_test, y_pred=y_pred)
    return accuracy_score(y_true=y_test, y_pred=y_pred)


df = create_training_dataframe(use_saved=True, clf="doc2vec")

transformer = FunctionTransformer(text_preprocessing)

t_pipeline = Pipeline(steps=[
    ("trans", transformer)
])

# model is named to base_model to resolve overwriting issues
base_model = Doc2Vec(
    vector_size=100,
    window=5,
    min_count=3,
    dm=1,
    workers=8,
    epochs=EPOCHS)

if not TRAIN_FINAL:
    # using StratifiedKFold for train_test_split
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True)
    accuracies = []
    split_index = 1
    for train_index, test_index in skf.split(df["content"], df["category"]):
        # applying the transform function on training and testing data
        X_train, X_test = t_pipeline.transform(df["content"][train_index]), t_pipeline.transform(df["content"][test_index])
        y_train, y_test = df["category"][train_index], df["category"][test_index]

        # see source[2]
        tagged_tr = [TaggedDocument(words=(str(X).split()), tags=[str(i)]) for i, X in enumerate(X_train)]
        tagged_test = [TaggedDocument(words=(str(X).split()), tags=[str(i)]) for i, X in enumerate(X_test)]

        if split_index == CV_SPLITS or TRAIN_ONCE:
            trained_model, lrc = train_model(base_model, tagged_tr, y_train, save_to_disk=SAVE_TO_DISK)
        else:
            trained_model, lrc = train_model(base_model, tagged_tr, y_train, save_to_disk=False)
        accuracy = test_model(trained_model, lrc, tagged_test, y_test)
        accuracies.append(accuracy)

        if TRAIN_ONCE:
            break

        split_index += 1

    print(accuracies)
    print("The mean accuracy is", np.mean(accuracies))
    print("Please note, this is the mean accuracy on each 500 word chunk. If your document constists of more than 500 words, it is probably way better.")

else:  # TRAIN_FINAL == TRUE; no testing, the whole dataset is used for training
    X_train = t_pipeline.transform(df["content"])
    y_train = df["category"]

    tagged_tr = [TaggedDocument(words=(str(X).split()), tags=[str(i)]) for i, X in enumerate(X_train)]
    train_model(base_model, tagged_tr, y_train, save_to_disk=SAVE_TO_DISK)
