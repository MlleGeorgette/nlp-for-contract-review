# Import models
# from pywebio.platform.flask import webio_view
# from pywebio import STATIC_PATH
# from flask import Flask, send_from_directory
import pywebio
from pywebio.input import *
from pywebio.output import *
from pywebio.session import run_js
import joblib
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from keras_preprocessing.sequence import pad_sequences
import time

# Load models
BinaryLabelModel = joblib.load('models/BinaryLabelModel_CNN.pkl')
MultiLabelModel = joblib.load('models/MultiLabelModel_CNN.pkl')
MultiLabelTokenizer = joblib.load('models/MultiLabelTokenizer.pkl')
BinaryLabelTokenizer = joblib.load('models/BinaryLabelTokenizer.pkl')
MultiLabelBinarizer = joblib.load('models/MultiLabelBinarizer_CNN.pkl')

# app = Flask(__name__)

# Information to be displayed
DisclaimerInfo = "Welcome to Contract Wiz! This application is for educational purposes only and does not constitute legal advice. Contract Wiz does not guarantee the accuracy or completeness of any information or analysis supplied. You should consult a qualified lawyer if you are seeking legal advice."
GeneralInfo = "The purpose of this application is to identify whether a sentence from a contract is a norm and if yes, to identify the relevant deontic label/s: PERMISSION/OBLIGATION/PROHIBITION."
InvalidInfo = "Sorry! This sentence cannot be classified as permission/obligation/prohibition because it is not a norm. Visit ABOUT to read more."

def NormInfo():
    popup('What is a norm?',[
        put_text("A norm refers to any sentence in a contract that describes the expected behaviour of one or more parties to the contract. It is typically expressed using deontic modalities (permission/obligation/prohibition) which are identifed by modal verbs (e.g. may, must, shall, will, may not, shall not)."),
        put_link('Read more', 'https://link.springer.com/content/pdf/10.1007/1-4020-3552-7_7.pdf', new_window=True),
        put_buttons(['Close'], onclick=lambda _: close_popup())
    ])

def PermissionInfo():
    popup('What is a permission?',[
        put_text("A permission is any behaviour that is allowed to be executed by a party to the contract. It is mainly expressed by the modal verb 'may' but other common verb formations include: can, shall be entitled to, shall be permitted to, will be entitled to."),
        put_buttons(['Close'], onclick=lambda _: close_popup())
    ])

def ObligationInfo():
    popup('What is an obligation?',[
        put_text("An obligation is a behaviour (usually a duty) that must be executed by a party. An obligation is typically identified by the verbs 'must' or 'shall'."),
        put_link('Read more', 'https://hs-legal.co.uk/services/individual-services/civil-litigation/contractual-obligation/', new_window=True),
        put_buttons(['Close'], onclick=lambda _: close_popup())
    ])

def ProhibitionInfo():
     popup('What is a prohibition?',[
        put_text("A prohibition is a behaviour that is forbidden and would result in a violation if executed by a party. It is typically written as the negation of an obligation or permission for example 'shall not' or 'may not'."),
        put_buttons(['Close'], onclick=lambda _: close_popup())
    ])
    

# Function to clean text (from pre-processing script)
# instantiate nltk lemmatizer
wnl = WordNetLemmatizer()

def CleanText(sentence):
     a = []
     tokens = word_tokenize(sentence)
     tokens = [token.lower() for token in tokens if token.isalpha()]
     for token in tokens:
          lem_word = wnl.lemmatize(token)
          a.append(lem_word)
     
     sentence = " ".join(a)
     return sentence

# Keras pre-processing
maxlen = 200
max_words = 2000
tokenizer_binary = BinaryLabelTokenizer
tokenizer_multilabel = MultiLabelTokenizer

# For binary classification - is sentence a norm?
def GetFeatures_Norm(text_series):
    sequences = tokenizer_binary.texts_to_sequences(text_series)
    return pad_sequences(sequences, maxlen=maxlen)

# For multilabel classification - what labels does a norm sentence have?
def GetFeatures_MultiLabel(text_series):
    sequences = tokenizer_multilabel.texts_to_sequences(text_series)
    return pad_sequences(sequences, maxlen=maxlen)

# Functions to validate user's acceptance of T&Cs
def ValidateDisclaimerInfo(opt):
    # print(opt)
    if not opt:
        return "You must accept the terms and conditions to proceed."

def ValidateGeneralInfo(opt):
    # print(opt)
    if not opt:
        return "I need to know you've got it before we proceed :)"


# Function to make prediction
def predict():
    put_image(open('gui/contract_wiz_banner.png', 'rb').read())
    put_text()

    # Terms and conditions
    disclaimer = checkbox(DisclaimerInfo, options=['Agree and proceed'], validate=ValidateDisclaimerInfo)

    # Information
    general_info = checkbox(GeneralInfo, options=['Got it!'], validate=ValidateGeneralInfo)

    # Collect data from user
    text = textarea("Insert sentence for review:", rows=5, placeholder="Place text here")

    # Loading bar
    with put_loading():
        time.sleep(6)

        # Clean input and predict norm
        cleaned_text = CleanText(text)
        norm_text = GetFeatures_Norm([cleaned_text])
        predict_norm = BinaryLabelModel.predict(norm_text)
        norm_probas = (predict_norm > 0.5).astype(int)
    
        # Predict tag if input is classified as a norm
        if norm_probas == [1]:
            deontic_text = GetFeatures_MultiLabel([cleaned_text])
            predict_tag = MultiLabelModel.predict(deontic_text)
            deontic_probas = (predict_tag > 0.5).astype(int)
            deontic_tag = MultiLabelBinarizer.inverse_transform(deontic_probas)

            # clean results (prediction is returned as a list of tuple)
            results = '\n'.join(deontic_tag[0]).upper()
            
            # Display results to user
            put_tabs([
                {'title': 'SUMMARY', 'content':
                put_table([
                    [span('Summary', col=2)],
                    ['Sentence', put_text(text)],
                    ['Tag/s', put_text(results)],
                    [span('DISCLAIMER: The above content is for general information purposes only. Please consult a qualified lawyer for legal advice.', col=2)]
                    ])
                },
                {'title': 'ABOUT', 'content':
                put_buttons(
                    ['Norm','Permission', 'Obligation', 'Prohibition'], 
                    onclick=[NormInfo, PermissionInfo, ObligationInfo, ProhibitionInfo])
                },
                {'title': 'MORE INFO', 'content': [
                put_text("For more information about this project, visit my GitHub profile:"),
                put_link('MlleGeorgette', 'https://github.com/MlleGeorgette/hello-world', new_window=True)
               ]},
            ])
            
        else:
            put_tabs([
                {'title': 'SUMMARY', 'content':
                put_text(InvalidInfo)
                },
                 {'title': 'ABOUT', 'content':
                put_buttons(
                    ['Norm','Permission', 'Obligation', 'Prohibition'], 
                    onclick=[NormInfo, PermissionInfo, ObligationInfo, ProhibitionInfo])
                },
                {'title': 'MORE INFO', 'content': [
                put_text("For more information about this project, visit my GitHub profile:"),
                put_link('MlleGeorgette', 'https://github.com/MlleGeorgette/hello-world', new_window=True)
               ]},
            ])
        
        put_text("*******************************************Thanks for using Contract Wiz!*******************************************")
        put_button("Reload", onclick=lambda: run_js('window.location.reload()'))
        put_image(open('gui/contract_wiz_footer.jpeg', 'rb').read())

# app.add_url_rule('/contractwiz', 'webio_view', webio_view(predict),
# methods=['GET', 'POST', 'OPTIONS'])

# app.run(host='Localhost', port=80, debug=True)

# visit http://Localhost/contractwiz to open the PyWebIO application

if __name__ == '__main__':
    pywebio.start_server(predict, port=8080, debug=True, remote_access=False)
