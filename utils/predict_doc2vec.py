import os
import numpy as np
import joblib

from settings import OUTPUT_PROBA, STORAGE_DIR

from utils.preprocessing import text_preprocessing
from utils.get_documents import create_prediction_dataframe

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

transformer = FunctionTransformer(text_preprocessing)

prediction_dataframe = create_prediction_dataframe(use_saved=False)

t_pipeline = Pipeline(steps=[
    ("trans", transformer)
])


model = Doc2Vec.load("storage/doc2vec.model")
lrc = joblib.load("storage/lrc.pkl")




tagged_X = [TaggedDocument(words=(str(X).split()), tags=[str(i)]) for i, X in enumerate(t_pipeline.transform(prediction_dataframe["content"]))]

X = np.array([model.infer_vector(tagged_X[i][0]) for i in range(len(tagged_X))])
y_pred = lrc.predict(X)
y_pred_probas = lrc.predict_proba(X)
print(y_pred_probas)
print("#" * 100)
print(y_pred)
prediction_dict = {}
for i, row in prediction_dataframe.iterrows():
    if row["filepath"] not in prediction_dict:
        prediction_dict[row["filepath"]] = []
    prediction_dict[row["filepath"]].append(y_pred_probas[i])

le = LabelEncoder()
le.classes_ = np.load(os.path.join(STORAGE_DIR, 'labelencoder_classes.npy'), allow_pickle=True)

for key, value in prediction_dict.items():
    if OUTPUT_PROBA:
        print(key, np.mean(value, axis=0))
    probability = np.mean(value, axis=0)
    prediction_dict[key] = {"predicted_category": le.inverse_transform([np.argmax(probability)])[0], "probability": probability}

print(prediction_dict)
