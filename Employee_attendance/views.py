from django.shortcuts import render, redirect
from .detection import FaceRecognition
# from .attendance_views import AttendanceManager
from .forms import *
from django.contrib import messages
from . employee_views import employee_home
from .admin_views import *
import wave
import os
import pyaudio
import re
import numpy as np
from sklearn.mixture import GaussianMixture
import pickle
from sklearn import preprocessing
import python_speech_features as mfcc
from scipy.io.wavfile import read
from EmployeeAttendance.settings import BASE_DIR
from django.contrib import messages

faceRecognition = FaceRecognition()
attendance = Attendance()


def calculate_delta(array):
    rows, cols = array.shape
    deltas = np.zeros((rows, 20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
                first = 0
            else:
                first = i-j
            if i+j > rows - 1:
                second = rows - 1
            else:
                second = i+j
            index.append((second, first))
            j += 1
        deltas[i] = (array[index[0][0]]-array[index[0][1]] +
                     (2 * (array[index[1][0]]-array[index[1][1]]))) / 10
    return deltas


def extract_features(audio, rate):
    mfcc_feat = mfcc.mfcc(audio, rate, 0.025, 0.01, 20,
                          appendEnergy=True, nfft=1103)
    mfcc_feat = preprocessing.scale(mfcc_feat)
    delta = calculate_delta(mfcc_feat)

    # combining both mfcc features and delta
    combined = np.hstack((mfcc_feat, delta))
    return combined


def train(directory_path, emp_id):
    print("training started")
    dest = "Employee_attendance/gmm_models/"
    count = 1

    for path in os.listdir(directory_path):
        path = os.path.join(directory_path, path)
        print("path:", path)

        features = np.array([])

        # reading audio files of speaker
        (sr, audio) = read(path)

        # extract 40 dimensional MFCC & delta MFCC features
        vector = extract_features(audio, sr)
        print("vector:", vector)
        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))

        # when features of 3 files of speaker are concatenated, then do model training
        if count == 3:
            gmm = GaussianMixture(
                n_components=16, covariance_type='diag', n_init=3)
            gmm.fit(features)

            # saving the trained gaussian model
            # pickle.dump(gmm, open(dest+'/'+ emp_id + '.gmm', 'wb'))
            with open(dest+'/' + emp_id + '.gmm', 'wb') as f:
                pickle.dump(gmm, f)
                print(emp_id + ' added successfully')

            features = np.asarray(())
            count = 0
        count = count + 1
        print("model trained successfully")


def recordvoice(request, emp_id):
    if request.method == 'POST':
        # Define the audio parameters
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 1024
        RECORD_SECONDS = 5

        # Create an instance of the PyAudio class
        audio = pyaudio.PyAudio()

        # Create the directory path if it doesn't exist
        directory_path = os.path.join(
            BASE_DIR, 'Employee_attendance', 'audio_dataset')
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        # Define the file path with the current timestamp
        file_path = os.path.join(directory_path, f'User.{emp_id}.wav')

        # Open a new stream to record audio from the user's microphone
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)

        # Record audio for RECORD_SECONDS seconds
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        # Stop and close the audio stream
        stream.stop_stream()
        stream.close()

        # Terminate the PyAudio instance
        audio.terminate()

        # Write the audio data to a WAV file
        wf = wave.open(file_path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        print("succesfully recorded")
        train(directory_path, emp_id)
        return redirect('home')

    return render(request, 'audio.html')


def recognisevoice(request):
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 3
    FILENAME = "./test.wav"

    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("recording...")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("finished recording")

    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # saving wav file
    waveFile = wave.open(FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    modelpath = "Employee_attendance\gmm_models"

    gmm_files = [os.path.join(modelpath, fname) for fname in
                 os.listdir(modelpath) if fname.endswith('.gmm')]

    models = [pickle.load(open(fname, 'rb'), encoding='latin1')
              for fname in gmm_files]

    speakers = [fname.split("/")[-1].split(".gmm")[0] for fname
                in gmm_files]

    if len(models) == 0:
        print("No Users in the Database!")
        return

    # read test file
    sr, audio = read(FILENAME)

    # extract mfcc features
    vector = extract_features(audio, sr)
    log_likelihood = np.zeros(len(models))

    # checking with each model one by one
    for i in range(len(models)):
        gmm = models[i]
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()

    pred = np.argmax(log_likelihood)
    identity = speakers[pred]

    # if voice not recognized than terminate the process
    if identity == 'unknown':
        print("Not Recognized! Try again...")
        return

    print("Recognized as - ", identity)

    emp_id = re.search(r'\d+', identity).group()

    return emp_id


def home(request):
    return render(request, 'home.html')


def register(request):
    if request.method == "POST":
        form = EmployeeForm(request.POST or None)
        if form.is_valid():
            emp_id = request.POST['emp_id']
            user = form.save(commit=False)
            user.user_type = 2
            user.save()
            Employee.objects.create(
                user=user, emp_id=request.POST['emp_id'], gender=request.POST['gender'], address=request.POST['address'])
            print(user)
            addFace(emp_id)
            return redirect('recordvoice', emp_id=emp_id)
            # return redirect('audio')
        else:
            messages.error(request, "Account registeration failed")
    else:
        form = EmployeeForm()
    return render(request, 'register.html', {'form': form})


def addFace(emp_id):
    try:
        print("IN add face", emp_id)
        faceRecognition.faceDetect(emp_id)
        print("detection done")
        faceRecognition.trainFace()
        print("trained")
    except:
        print("Error occured")
