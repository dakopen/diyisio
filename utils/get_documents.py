import os
import pandas as pd
import numpy as np
import logging
import warnings

import PyPDF2
from docx2python import docx2python

from settings import PREDICT_DIR, STORAGE_DIR, DOCUMENTS_DIR, MAX_WORDS

from sklearn.preprocessing import LabelEncoder


def get_file_names(DIR, only_file_name=False):
    """Retrieves all filepaths for documents in the following structure:
    documents/
    ├─ class1 (e.g. manuals)/
    │  ├─ document1
    ├─ class2 (e.g. cover letters and cv)/
    │  ├─ document1
    │  ├─ document2
    │  ├─ document3
    ├─ class3 (e.g. invoices)/
    │  ├─ [EMPTY --> ignored]
    ├─ classlist (e.g. subjects)/
    │  ├─ class4 (e.g. math)/
    │  │  ├─ document1
    │  ├─ class5 (e.g. biology)/
    │  │  ├─ document1
    ├─ .../
    ├─ ...
    returns dictionary of the following type:
    {'class1': ['BASE_DIR/documents/class1/document1.pdf'],
    'class2': ['BASE_DIR/documents/class2/document1.pdf', 'BASE_DIR/documents/class2/document2.pdf',
                'BASE_DIR/documents/class2/document3.pdf'],
    'class4': ['BASE_DIR/documents/class4/document1.pdf']
    'class5': ['BASE_DIR/documents/class5/document1.pdf']}

    Please note: empty folders are completely ignored.
    If there was a document (that is not in a subfolder) in 'classlist', 'classlist' would be listed as well.
    """
    file_list = {}
    folderlist = ([x for x in os.walk(DIR)])

    for folder in folderlist:
        if len(folder[2]) >= 1:
            logging.debug("Collecting folder " + os.path.basename(os.path.normpath(folder[0])))

            file_list[os.path.basename(os.path.normpath(folder[0]))] = []
            for filepath in folder[2]:
                if not only_file_name:
                    file_list[os.path.basename(os.path.normpath(folder[0]))].append(os.path.join(folder[0], filepath))
                else:
                    file_list[os.path.basename(os.path.normpath(folder[0]))].append(filepath)
    return file_list


def get_file_content(filepath):
    """Extracts the text of every page in the PDF-file and joins them."""
    logging.debug("Extracting file content from " + filepath)

    try:
        filename, file_extension = os.path.splitext(filepath)
        if str(file_extension).lower() == ".pdf":  # PDF
            extracted_text = []
            pdffileobj = open(filepath, 'rb')
            pdfreader = PyPDF2.PdfFileReader(pdffileobj)

            page_count = pdfreader.numPages

            for x in range(page_count):
                pageobj = pdfreader.getPage(x)
                page_text = pageobj.extractText()
                extracted_text.append(page_text)
            return " ".join(extracted_text)

        elif str(file_extension).lower() == ".docx":  # Word
            document = docx2python(filepath)
            return document.text

        else:
            return np.nan  # Return nan content. Row is skipped later.
    except Exception as e:
        warnings.warn(f"Error on file extraction {filepath}: {str(e)}")
        return np.nan

def create_training_dataframe(use_saved=True, clf: str = "doc2vec"):
    """Creates and returns pandas Dataframe build from the 'documents' directory. Contains category and prediction.
    Content may dispense over multiple rows."""
    try:
        if not use_saved:  # if the data has changed or should be reloaded
            raise FileNotFoundError

        return pd.read_pickle(os.path.join(STORAGE_DIR, "training_dataframe.pkl"))

    except FileNotFoundError:
        filenames = get_file_names(DOCUMENTS_DIR)

        data_for_df = []

        for category, files in filenames.items():
            for file in files:
                data_for_df.append([category, file])
        df = pd.DataFrame(data_for_df, columns=["category", "filepath"])

        le = LabelEncoder().fit(df['category'])

        df['category'] = le.transform(df['category'])  # transforms a class name to the corresponding encoded label

        np.save(os.path.join(STORAGE_DIR, f'labelencoder_classes_{clf}.npy'), le.classes_)  # see source[1]

        df["content"] = df.apply(lambda row: get_file_content(row["filepath"]), axis=1)
        df = df.drop(['filepath'], axis=1)
        df.dropna(subset=['content'])

        changed = True
        while changed:
            changed = False
            for index, row in df.iterrows():
                if len(str(row["content"]).split()) > MAX_WORDS:
                    original_row = row.copy()
                    # make sure that e.g. 510 words are splittet equally instead of 500 and 10
                    if len(str(original_row["content"]).split()) > 2 * MAX_WORDS:
                        df.at[index, "content"] = " ".join(str(original_row["content"]).split()[:MAX_WORDS])
                        row_append = [original_row["category"], " ".join(str(original_row["content"]).split()[MAX_WORDS:])]
                        df = pd.concat([df, pd.DataFrame([row_append], columns=["category", "content"])],  ignore_index=True)
                    else:
                        df.at[index, "content"] = " ".join(str(original_row["content"]).split()[:int(len(str(original_row["content"]).split())/2)])
                        row_append = [original_row["category"], " ".join(str(original_row["content"]).split()[int(len(str(original_row["content"]).split())/2):])]
                        df = pd.concat([df, pd.DataFrame([row_append], columns=["category", "content"])],  ignore_index=True)

                    changed = True

        df.reset_index(drop=True, inplace=True)
        df.to_pickle(os.path.join(STORAGE_DIR, "training_dataframe.pkl"))  # save the dataframe for later use
        return df


def create_prediction_dataframe(use_saved=False):
    """Creates and returns pandas Dataframe build from the 'predict' directory. Contains content and filepath.
    Content may spread over multiple rows."""
    try:
        if not use_saved:  # if the data has changed or should be reloaded
            raise FileNotFoundError

        return pd.read_pickle(os.path.join(STORAGE_DIR, "prediction_dataframe.pkl"))

    except FileNotFoundError:
        filenames = get_file_names(PREDICT_DIR)
        df = pd.DataFrame(list(filenames.values())[0], columns=["filepath"])
        print(df)
        df["content"] = df.apply(lambda row: get_file_content(row["filepath"]), axis=1)
        df.dropna(subset=['content'])

        # KEEP THE FILEPATH AS PRIMARY KEY (since the content may be spread)!!
        changed = True
        while changed:
            changed = False
            for index, row in df.iterrows():
                if len(str(row["content"]).split()) > MAX_WORDS:
                    original_row = row.copy()

                    # make sure that e.g. 510 words are splittet equally instead of 500 and 10
                    if len(original_row["content"].split()) > 2 * MAX_WORDS:
                        df.at[index, "content"] = " ".join(original_row["content"].split()[:MAX_WORDS])
                        row_append = [row["filepath"], " ".join(original_row["content"].split()[MAX_WORDS:])]
                        df = pd.concat([df, pd.DataFrame([row_append], columns=["filepath", "content"])],  ignore_index=True)
                    else:
                        df.at[index, "content"] = " ".join(original_row["content"].split()[:int(len(original_row["content"].split())/2)])
                        row_append = [row["filepath"], " ".join(original_row["content"].split()[int(len(original_row["content"].split())/2):])]
                        df = pd.concat([df, pd.DataFrame([row_append], columns=["filepath", "content"])],  ignore_index=True)

                    changed = True

        df.reset_index(drop=True, inplace=True)
        df.to_pickle(os.path.join(STORAGE_DIR, "prediction_dataframe.pkl"))  # save the dataframe for later use
        return df
