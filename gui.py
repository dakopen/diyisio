import PySimpleGUI as sg
from settings import *
from utils.get_documents import get_file_names
HEIGHT = 600


def get_train_files_column():
    file_names_dict = get_file_names(DOCUMENTS_DIR, only_file_name=True)
    file_column = []
    for category, filename in file_names_dict.items():
        file_column.append([sg.Text(f"{category}:", font=("Arial", 12, "bold"), background_color="green")])
        for file in filename:
            file_column.append([sg.Text(f"◉ {file}")])

    return file_column

def get_predict_files_column():
    file_names_dict = get_file_names(PREDICT_DIR, only_file_name=True)
    file_column = []
    for category, filename in file_names_dict.items():
        file_column.append([sg.Text(f"{category}:", font=("Arial", 12, "bold"), background_color="orange")])
        for file in filename:
            file_column.append([sg.Text(f"◉ {file}")])

    return file_column

toggle_column = [[sg.Text("Train/Predict:"), sg.Button('Train Files', size=(10, 1), button_color=('white', 'green'), key='_TRAIN_PREDICT_')]]


col1 = [[sg.Text("Train on these files")],
        [sg.Column(get_train_files_column(), element_justification='l', scrollable=True, size=(380, HEIGHT-100), visible=True, key="training_files_column")],
        [sg.Column(get_predict_files_column(), element_justification='l', scrollable=True, size=(380, HEIGHT - 100), visible=False, key="prediction_files_column")]
        ]
col2 = [[sg.Text("Controls")],
        [sg.Column(toggle_column, element_justification='c', size=(400, HEIGHT))]]
col3 = [[sg.Text("Console and start training button")]]

#layout = [  [sg.Text('Some text on Row 1')],
#            [sg.Text('Enter something on Row 2'), sg.InputText()],
#            [sg.Button('Ok'), sg.Button('Cancel')] ]

layout = [[sg.Column(col1, element_justification='c', size=(400, HEIGHT)), sg.Column(col2, element_justification='c', size=(400, HEIGHT))
          , sg.Column(col3, element_justification='c', size=(400, HEIGHT))]]

train_files = True
# Create the Window
window = sg.Window('diyisio', layout, scaling=2, size=(1400, HEIGHT))

# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel':  # if user closes window or clicks cancel
        break

    if event == '_TRAIN_PREDICT_':
        train_files = not train_files
        window.Element("training_files_column").Update(visible=(True, False)[train_files])
        window.Element("prediction_files_column").Update(visible=(False, True)[train_files])
        window.Element('_TRAIN_PREDICT_').Update(('Predict files', 'Train on files')[train_files], button_color=(('white', ('orange', 'green')[train_files])))



window.close()
