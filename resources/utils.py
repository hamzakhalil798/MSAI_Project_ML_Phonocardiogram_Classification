import warnings
import librosa
import pandas as pd
import numpy as np
import python_speech_features as mf

def feature_1(file):
    audio,sr = librosa.load(file)
    zcr = librosa.zero_crossings(audio)
    zcr = sum(zcr)
    data = pd.DataFrame([zcr],columns=['A'])
    return data

def feature_2(file):
    audio,sr = librosa.load(file)

    #mfcc=np.mean(librosa.feature.mfcc(audio,sr = sr,n_mfcc=12).T,axis=0)


    mfcc_feature =np.mean(mf.mfcc(audio,sr, 0.025, 0.01,12,nfft = 1200, appendEnergy = True),axis = 0)
    #mfcc_feature = preprocessing.scale(mfcc_feature)
    mfcc = pd.DataFrame(mfcc_feature)
    #df1 = pd.DataFrame(mfcc)
    #df1 = df1.T
    mfcc = mfcc.T
    return mfcc


def feature_3(file):
    try:
        # Load the audio file
        audio, sr = librosa.load(file)

        # Extract chroma feature
        chromagram = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=512),axis=1)

        # Create a DataFrame from the chroma feature
        cr = pd.DataFrame(chromagram)

        cr=cr.T

        return cr
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return None  # Return None in case of an error









