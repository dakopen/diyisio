import os
import warnings
import logging
logging.basicConfig(level=logging.CRITICAL)  # set to DEBUG to see if anything is working
logger = logging.getLogger("PyPDF2")
logger.setLevel(logging.ERROR)

BASE_DIR = os.getcwd()
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
PREDICT_DIR = os.path.join(BASE_DIR, "predict")

USE_TRAINED = True


def get_least_populated_category():
    folderlist = ([x for x in os.walk(DOCUMENTS_DIR)])

    min_file_count = []
    for folder in folderlist:
        i = 0
        for file in folder[2]:
            if str(file).lower().endswith(".docx") or str(file).lower().endswith(".pdf"):
                i += 1
        if i > 0:
            min_file_count.append(min_file_count)
    return min(min_file_count)


# see source[3] for the difference between stemming and lemmatization
TEXT_PREPROCESSING = "lemma"  # "lemma" (recommended) for lemmatization; "stem" for stemming and None for none.

MAX_WORDS = 500

CV_SPLITS = min(([int(get_least_populated_category()/2), 15]))  # set the CV_SPLITS to half of the least populated category; max: 15

if CV_SPLITS < 5:
    warnings.warn("Please provide at least 10 examples per category")

EPOCHS = 100  # (100 is recommended), 50 will do fine too. The EPOCHS only apply to the doc2vec training.

TRAIN_FINAL = False
TRAIN_ONCE = False
SAVE_TO_DISK = any([TRAIN_FINAL, False])  # set True to False in order to disable SAVE_TO_DISK if TRAIN_FINAL is False

OUTPUT_PROBA = False  # set to True, if you're interested in the probabilities
