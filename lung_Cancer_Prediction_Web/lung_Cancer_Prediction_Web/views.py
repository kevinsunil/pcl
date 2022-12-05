from django.shortcuts import render
import pickle
import os.path
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
class_a_path = os.path.join(BASE_DIR, "lung_cancer_ml_model.sav")
# our home page view


def home(request):
    return render(request, 'index.html')


def getPredictions(GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN):

    model = pickle.load(open(class_a_path, "rb"))
    print("111111111")
    input_arr = [GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE, FATIGUE,
                 ALLERGY, WHEEZING, ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN]
    input_asnparray = np.asarray(input_arr)

    input_reshaped = input_asnparray.reshape(1, -1)
    prediction = model.predict(input_reshaped)

    print("prediction", prediction)
    if prediction == 0:
        return "Lucky you get to live another day"
    elif prediction == 1:
        return "Work on your self"
    else:
        return "error"


def result(request):
    GENDER = int(request.GET['gender'])
    AGE = int(request.GET['age'])
    SMOKING = int(request.GET['smoking'])
    YELLOW_FINGERS = int(request.GET['fingure'])
    ANXIETY = int(request.GET['anxiety'])
    PEER_PRESSURE = int(request.GET['peerPressure'])
    CHRONIC_DISEASE = int(request.GET['chronicDisease'])
    FATIGUE = int(request.GET['fatigue'])
    ALLERGY = int(request.GET['allergy'])
    WHEEZING = int(request.GET['wheezing'])
    ALCOHOL_CONSUMING = int(request.GET['alcoholConsumption'])
    COUGHING = int(request.GET['coughing'])
    SHORTNESS_OF_BREATH = int(request.GET['shortnedOfBreath'])
    SWALLOWING_DIFFICULTY = int(request.GET['swallowingDifficulty'])
    CHEST_PAIN = int(request.GET['chestPain'])

    result = getPredictions(GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE, FATIGUE,
                            ALLERGY, WHEEZING, ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN)
    return render(request, 'result.html', {'result': result})
