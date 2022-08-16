import os
import numpy as np
import joblib

from settings import OUTPUT_PROBA, STORAGE_DIR, CUSTOM_NAME_SUFFIX

from utils.get_documents import create_prediction_dataframe

from sklearn.preprocessing import LabelEncoder


class Predict_Votingclassifier():
    def __init__(self):
        prediction_dataframe = create_prediction_dataframe(use_saved=False)

        lrc = joblib.load(os.path.join(STORAGE_DIR, f'voting_classifier{CUSTOM_NAME_SUFFIX}.pkl'))

        y_pred = lrc.predict(prediction_dataframe["content"])
        y_pred_probas = lrc.predict_proba(prediction_dataframe["content"])

        prediction_dict = {}
        for i, row in prediction_dataframe.iterrows():
            if row["filepath"] not in prediction_dict:
                prediction_dict[row["filepath"]] = []
            prediction_dict[row["filepath"]].append(y_pred_probas[i])

        le = LabelEncoder()
        le.classes_ = np.load(os.path.join(STORAGE_DIR, f'labelencoder_classes_votingclassifier{CUSTOM_NAME_SUFFIX}.npy'), allow_pickle=True)

        for key, value in prediction_dict.items():
            if OUTPUT_PROBA:
                print(key, np.mean(value, axis=0))
            probability = np.mean(value, axis=0)
            prediction_dict[key] = {"predicted_category": le.inverse_transform([np.argmax(probability)])[0], "probability": probability}

        print(prediction_dict)
        self.prediction_dict = prediction_dict

    def get_prediction_dict(self):
        return self.prediction_dict
