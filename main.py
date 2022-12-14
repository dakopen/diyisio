import os
from settings import USE_TRAINED, STORAGE_DIR, CUSTOM_NAME_SUFFIX, MODEL
from utils.predict_doc2vec import Predict_doc2vec
from utils.train_doc2vec import Train_doc2vec
from utils.predict_votingclassifier import Predict_Votingclassifier
from utils.train_votingclassifier import Train_Votingclassifier


class Main:
    def __init__(self):

        if MODEL == "Doc2Vec":
            if USE_TRAINED:
                # Double check that they all exist
                if os.path.exists(os.path.join(STORAGE_DIR, f"labelencoder_classes_doc2vec{CUSTOM_NAME_SUFFIX}.npy")) and \
                        os.path.exists(os.path.join(STORAGE_DIR, f"doc2vec{CUSTOM_NAME_SUFFIX}.model")) and \
                        os.path.exists(os.path.join(STORAGE_DIR, f"lrc{CUSTOM_NAME_SUFFIX}.pkl")):
                    print("Start prediction...")
                    self.prediction = Predict_doc2vec()

                else:
                    raise Exception("USE_TRAINED is set to TRUE, but corresponding files don't exist.")
            else:
                print("Start training...")
                self.training = Train_doc2vec()
        elif MODEL == "Voting Classifier":
            if USE_TRAINED:
                # Double check that they all exist
                if os.path.exists(os.path.join(STORAGE_DIR, f"labelencoder_classes_votingclassifier{CUSTOM_NAME_SUFFIX}.npy")) and \
                        os.path.exists(os.path.join(STORAGE_DIR, f"voting_classifier{CUSTOM_NAME_SUFFIX}.pkl")):
                    print("Start prediction...")
                    self.prediction = Predict_Votingclassifier()

                else:
                    raise Exception("USE_TRAINED is set to TRUE, but corresponding files don't exist.")
            else:
                print("Start training...")
                self.training = Train_Votingclassifier()

    def get_prediction_dict(self):
        return self.prediction.get_prediction_dict()

    def get_accuracies(self):
        return self.training.get_training_accuracies()
