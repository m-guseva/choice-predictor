import pandas as pd
import streamlit as st
import numpy as np
from xgboost import XGBRegressor
from itertools import chain
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(layout="wide")


################################
# Constants

# How long should model learn in the beginning before making first prediction:
initSeqLen =18
# Interval in which the  model should be retrained
interValToTest = 1
sequence = []
button_sequence = []
sequence_str= []
perc_correctPredictions = []


################################
# Initialize session state variables
if 'button_sequence' not in st.session_state:
    st.session_state.button_sequence = []
if 'which_model' not in st.session_state:
    st.session_state.which_model = []
if 'perc_correctPredictions' not in st.session_state:
    st.session_state.perc_correctPredictions = []
if 'featureList' not in st.session_state:
    st.session_state.featureList = []
if 'predictionList' not in st.session_state:
    st.session_state.predictionList = []

################################
# Functions

def walkForwardValidation(supervised_values, which_model):
    '''
    Trains the predictive model using walk-forward validation
    Walk-forward validation:
    It fits the model based on train dataset. With each iteration in t, one row
    is added to the train dataset (=history) to fit the model to a dataset with one more observation 
    in other words, it takes a chunk of validation set with every iteration)
    '''    
    # Determine change index (necessary to know size of initial train chunk, because if target is only 1s or 0s model won't run)
    change_index = np.where(np.diff(supervised_values[:, -1]) != 0)[0] 
    train, validate = split_train_validate(supervised_values, change_index[0]+2)
    history = train.copy()
    predictions = []

    # Walk-forward validation
    for t in range(len(validate)):     
        # Get appropriate validation row to validate predictions:
        X_validate = validate[t, 0:-1].reshape(1, -1)
        y_validate = validate[t,-1].reshape(1,-1)
        # Configure model
        model = configModel(which_model)
        # Split in predictor and target
        X_train, y_train = history[:, 0:-1], history[:, -1] 
        # Fit model
        model.fit(X_train, y_train)
        # Make predictions for the current time step
        yhat = model.predict(X_validate)
        predictions.append(yhat[0])
        # Update the history with the observed value
        history = np.vstack((history, validate[t, :])) #append validation set row to history

    # Evaluate the model accuracy
    if which_model == 'logreg':
        validation_accuracy = np.sum(predictions == validate[:, -1]) / len(predictions)
    elif which_model =='xgboost' or which_model =='randomForest':
        validation_accuracy = np.sum(transform_predictions(predictions) == validate[:, -1]) / len(predictions)
    return model,validation_accuracy

def split_train_validate(supervised_values, len_initValSet):
    n_test = len(supervised_values)-len_initValSet
    train, validate = supervised_values[0:-n_test, :], supervised_values[-n_test:, :]
    return train, validate

def transform_predictions(predictions):
    predictions = [0 if pred < 0.5 else 1 for pred in predictions]
    return predictions

def split_sequence_to_df(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x = sequence[i:end_ix]
        seq_y = sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    X_df = pd.DataFrame(X, columns=[f'lag_{i}' for i in range(n_steps)])
    y_df = pd.DataFrame(y, columns=['output'])
    return pd.concat([X_df, y_df], axis=1)

def gatherChoiceSequence(symbol):
    ''' Takes clicked button as input and outputs a concatenated sequence of choice so far as integers'''
    sequence = st.session_state.button_sequence.append(symbol)
    sequence_str = ' '.join(map(str, st.session_state.button_sequence))
    sequence = [int(x) for x in sequence_str if x.strip().isdigit()]
    return sequence

def configModel(which_model):
    if which_model == 'logreg':
        model = LogisticRegression(max_iter=2000)
    elif which_model =='xgboost':
        model = XGBRegressor(n_estimators =1000, learning_rate = 0.05, n_jobs = 1)
    elif which_model == 'randomForest':
        model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=None)
    return model

def predictNextChoice(sequence,  which_model,features):
    ''' Takes sequence and features parameter and outputs the predicted next choice'''
    #1. Create supervised sequence
    supervised_sequence = split_sequence_to_df(sequence, features).values
    # 2. Split into features and target
    X_train, y_train = supervised_sequence[:, 0:-1], supervised_sequence[:, -1]
    #3. Fit model to X_train and y_train and create prediction
    model = configModel(which_model)
    model.fit(X_train, y_train)
    last_choices = np.array(sequence [-features:]).reshape(1,features)
    predicted_choice = transform_predictions(model.predict(last_choices))
    return predicted_choice


def searchFeatureSpace(sequence, feature_space):
    '''Takes sequence and tests different combinations of features, then outputs the best feature 
    (which is the one with the highest validation accuracy)'''
    validation_accuracies = {}
    for features in feature_space:
        # Reformulate dataset into supervised lagged values set
        supervised_values = split_sequence_to_df(sequence, features).values
        # Run model
        model,validation_accuracy = walkForwardValidation(supervised_values, which_model)
        validation_accuracies[features] = validation_accuracy
    bestFeature = max(validation_accuracies, key=validation_accuracies.get)
    return bestFeature

def whichFeatures(sequence):
    '''Calculates the best number of features to use in model. It takes the first choices up to initSeqLen 
     and calculates initBestFeature. As soon as our sequence grows to a multiplicative of initSeqLen, it recalculates the
      best number of features and returns the new one. The calculated features are saved in a st.session variable so
       that they survive each rerunning of the script. '''
    valuesToTest = [1,2,3,4,5]
    initBestFeature = []
    featureToUse = []
    if len(sequence) == initSeqLen:
        initBestFeature = searchFeatureSpace(sequence, valuesToTest)
        st.session_state.featureList = [initBestFeature]
        featureToUse = initBestFeature
    # Retrain after every 5 steps
    if len(sequence) % interValToTest == 0 and len(sequence) > initSeqLen:
        bestFeature = searchFeatureSpace(sequence, valuesToTest)
        st.session_state.featureList.append(bestFeature) # save feature in list
    if len(sequence)>initSeqLen:
        featureToUse = st.session_state.featureList[-1]
    return featureToUse

def procedure(sequence):
    ''' Takes sequence, determines optimal parameter (feature) and makes prediction'''
    # Determine how many features to use:
    n_features = whichFeatures(sequence) 
    prediction = []
    # Make prediction:
    if len(sequence)>initSeqLen:
        prediction = predictNextChoice(sequence, which_model,features=n_features)
        st.session_state.predictionList.append(prediction)
        print("N_features: ", n_features) # for internal debugging
    else:
        st.session_state.predictionList.append([])
    return prediction

def uiElements(sequence):
    # Display user's choice:
    if len(sequence) <1:
        col2.write(" ")        
        if (not left_button) and (not right_button):
            with col2:
                modelChoice_radio = st.radio("1️⃣ Pick an algorithm:", ["xgboost", "logreg", "randomForest"], horizontal=False)
                st.session_state.which_model = modelChoice_radio
        else:
            st.write(" ")
        col2.write("2️⃣ Then start making your choices by pressing the buttons on the right!")
    else:
        with col2:
            st.metric(label="▫️YOUR CHOICE▫️", value=str(sequence[-1]))
    if len(sequence)<=initSeqLen+1:
        col1.write("💬 Continue making choices, so I can learn!")
        col1.markdown("Number of choices before I begin predicting:  " + ":red["+str(initSeqLen+2-len(sequence))+"]")
    else:
        # Display computer's prediction:
        col2.metric(label="✨WHAT I PREDICTED ✨", value=int(st.session_state.predictionList[-2][0]))
        displayStats()
    return
    

def displayStats():
    '''
    Here we take all the values from the session state variable predictionList, flatten it and combine it with the sequence
    list into a df. In this df we calculate pred-vals and count how many 0s we get, divide it by the length and we get
    the percentage of correct predictions. This value is stored in the session state variable perc_correctPredictions
    and plotted in a line chart.
    '''
    with col6:
        # Flatten prediction list session state variable:
        plist = [int(item) for item in chain(*st.session_state.predictionList)]
        # Create df with predicted and chosen values. 
        data = pd.DataFrame({"pred":plist[:-1],"vals":sequence[initSeqLen+1:]})
        # Calculate percent of cases where pred-vals == 0 (i.e. correct predictions from the computer)
        perc_correct = ((data.pred-data.vals == 0).sum())/len(data.pred)
        # Update session state variable per_correctPredictions
        st.session_state.perc_correctPredictions.append(perc_correct)
        st.write("% correct predictions after every trial ⬇️")
        st.line_chart(st.session_state.perc_correctPredictions)
        
    with col2:
        if len(st.session_state.perc_correctPredictions)>=2:
            st.divider()
            delta = st.session_state.perc_correctPredictions[-1]-st.session_state.perc_correctPredictions[-2]
            st.metric(label="% correct predictions", value= np.round(perc_correct,2), delta=np.round(delta, 2))
    return

################################
# Front end
st.title("Choice predictor")
string = '''
1. Choose the algorithm to use for prediction. If you don't choose anything, then the default algorithm will be xgboost. 
2. Start making your sequence of choices by pressing the buttons 1 or 0 many times. Make sure to vary
your choices because if you  use only :red[one] of the buttons, the algorithm will throw an error!
3. After the algorithm has learned, it will start predicting your choices, the prediction accuracy will be shown on the right of the screen.
'''
st.markdown(string)

# Create layout
col1,col2, col3, col4, col5,col6 = st.columns([0.3,0.2,0.1,0.1,0.1,0.3])
col1.image("fortuneTeller.png")
# Buttons
left_symbol = "1"
right_symbol = "0"

# Make some space above buttons for better layout
for _ in range(3):  
    for col in [col3, col4]:
        col.write(" ")
        col.write(" ")

left_button = col3.button(left_symbol,key=1,use_container_width=True)
right_button = col4.button(right_symbol,key=0,use_container_width=True)


if left_button:
    sequence = gatherChoiceSequence(left_symbol)
elif right_button:
    sequence = gatherChoiceSequence(right_symbol)

# Handle case where person presses only one button
if (len(sequence) >= initSeqLen) & (len(set(sequence[:initSeqLen])) == 1):
    with col2:
        st.error("It seems like you have chosen only one of the options! 😱  The algorithm needs you to make choices between both 1 and 0! ☝🏻 Please refresh the page to start anew.", icon="‼️")
        st.stop()

# Which model to use ('logreg', 'xgboost' or  'randomForest')
which_model  = st.session_state.which_model
prediction = procedure(sequence)
uiElements(sequence)
    

################################
# string of sequence and prediction list for display in terminal
print("This is the choice sequence:")
print(str(sequence))
print("This is the list of predictions:")
print(str(st.session_state.predictionList))
print("This is the feature list")
print(st.session_state.featureList)

################################





