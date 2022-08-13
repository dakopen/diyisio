import os
from settings import USE_TRAINED, STORAGE_DIR

if USE_TRAINED:
    if os.path.exists(os.path.join(STORAGE_DIR, "labelencoder_classes.npy")) and \
            os.path.exists(os.path.join(STORAGE_DIR, "doc2vec.model")) and \
            os.path.exists(os.path.join(STORAGE_DIR, "lrc.pkl")):
        import utils.predict_doc2vec
    else:
        raise Exception("USE_TRAINED is set to TRUE, but corresponding files don't exist.")
else:
    import utils.train_doc2vec  # use import statement to execute the file and train the model
