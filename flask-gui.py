import os
import time
import webbrowser
import threading
from flask import Flask, render_template, redirect, request, jsonify
import json
BASE_DIR = os.getcwd()
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
PREDICT_DIR = os.path.join(BASE_DIR, "predict")

on_off_dict = {"on": "True", "off": "False"}
thread_object = None

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

class Run_with_config_Thread(object):
    def __init__(self,):
        thread = threading.Thread(target=self.run)
        thread.daemon = True  # Daemonize thread
        thread.start()

    def run(self):
        global thread_object
        from main import Main  # import here in order to have the settings not confused
        thread_object = Main()
        return thread_object


def get_file_names(DIR, only_file_name=False):
    file_list = {}
    folderlist = ([x for x in os.walk(DIR)])

    for folder in folderlist:
        if len(folder[2]) >= 1:

            file_list[os.path.basename(os.path.normpath(folder[0]))] = []
            for filepath in folder[2]:
                if not only_file_name:
                    file_list[os.path.basename(os.path.normpath(folder[0]))].append(os.path.join(folder[0], filepath))
                else:
                    file_list[os.path.basename(os.path.normpath(folder[0]))].append(filepath)
    return file_list


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True, template_folder="template")
    app.config.from_mapping(
        SECRET_KEY='no_use',
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/')
    def first_open():
        return redirect("/train")

    @app.route('/predict', endpoint="predict")
    def predict():
        file_names_dict = get_file_names(PREDICT_DIR, only_file_name=True)

        return render_template(os.path.join("predict.html"), context={"p_text": "Predict"}, files=file_names_dict, directory=PREDICT_DIR)

    @app.route('/predict_submit', methods=['POST'])
    def receive_predict_params():
        global thread_object
        thread_object = None
        os.environ["USE_TRAINED"] = "True"
        os.environ["OUTPUT_PROBA"] = on_off_dict[request.form.get("output_probabilities", "off")]

        CUSTOM_NAME_SUFFIX = str(request.form.get("custom_name_suffix", ""))
        if CUSTOM_NAME_SUFFIX:
            CUSTOM_NAME_SUFFIX = "_" + CUSTOM_NAME_SUFFIX
        os.environ["CUSTOM_NAME_SUFFIX"] = CUSTOM_NAME_SUFFIX

        Run_with_config_Thread()

        return redirect("/displaydata")


    @app.route('/train_submit', methods=['POST'])
    def receive_train_params():
        global thread_object
        thread_object = None
        os.environ["USE_TRAINED"] = "False"
        os.environ["TRAIN_FINAL"] = on_off_dict[request.form.get("train_final", "off")]
        os.environ["TRAIN_ONCE"] = on_off_dict[request.form.get("train_once", "off")]
        os.environ["SAVE_TO_DISK"] = on_off_dict[request.form.get("save_to_disk", "off")]
        os.environ["EPOCHS"] = request.form.get("epochs", "100")

        CUSTOM_NAME_SUFFIX = str(request.form.get("custom_name_suffix", ""))
        if CUSTOM_NAME_SUFFIX:
            CUSTOM_NAME_SUFFIX = "_" + CUSTOM_NAME_SUFFIX
        os.environ["CUSTOM_NAME_SUFFIX"] = CUSTOM_NAME_SUFFIX

        Run_with_config_Thread()

        return redirect("/displaydata")

    @app.route('/train', endpoint="train")
    def predict():
        file_names_dict = get_file_names(DOCUMENTS_DIR, only_file_name=True)

        return render_template(os.path.join("train.html"), context={"t_text": "Train"}, files=file_names_dict, directory=DOCUMENTS_DIR)


    @app.route('/displaydata')
    def displaydata():
        global thread_object
        if thread_object is None:
            return render_template(os.path.join("not_ready_yet.html"))
        return render_template(os.path.join("response_table.html"), prediction=thread_object.get_prediction_dict())



    return app
if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")
    app = create_app()
    app.run(debug=False)
