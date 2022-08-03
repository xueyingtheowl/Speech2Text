### PART I: Speech Recognition (Audio to Text) Pipeline

#install libraries
# Jupyter Notebook users: put %%bash in front of the below line before running
# other python IDE users (e.g., Atom, PyCharm, VSCode), run the below line in your terminal

pip3 install SpeechRecognition pydub

# import libraries
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import glob
import os
import string
import shutil
import PurePath
from string import digits
from string import whitespace

# assign the recognizer a short name
r = sr.Recognizer()

# get the file list of your recorded audios
listOFaudios = glob.glob('/users/Wubabear/Desktop/recordedAudio/*.wav')

for i, audioPath in enumerate(listOFaudios, start = 1):
    sound = AudioSegment.from_wav(audioPath)  
    # split audio sound where silence is 500 miliseconds or more to get chunks
    chunks = split_on_silence(sound,
        min_silence_len = 500,           # adjustable
        silence_thresh = sound.dBFS-14,  # adjustable
        keep_silence=500)                # adjustable
    
    # make a directory to store incoming segmented chunks
    basePath = "/users/Wubabear/recordedAudio/123/"
    createPath = basePath + "folder_" + str(i)
    audioChunkFolder = os.mkdir(createPath)
    whole_text = ""
    
    # process each chunk 
    for q, audio_chunk in enumerate(chunks, start=1):
        chunk_filename = os.path.join(createPath, "chunk"+str(q)+"wav")
        audio_chunk.export(chunk_filename, format="wav")

        with sr.AudioFile(chunk_filename) as source:    # recognize the chunk            
            audio_listened = r.record(source)
            try:                                           
                text = r.recognize_google(audio_listened)
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}. "
                whole_text += text

    with open (os.path.join('/users/Wubabear/Desktop/script', "audio"+str(i)+".txt"), 'w') as f:
        f.write(whole_text)


### PART II: Preprocess Files to Meet Prerequisites of MFA. 
### (Skip part II and go to part III directly if you don't use the data in SP02 folder and use 
###  the autorecognized text )

## Step1: delete unused files and convert file formats in SP02_Mock folder

# delete unused files--uncorrected transcripts
fileList0 = glob.glob("/users/Wubabear/Desktop/SP02_Mock/*t.txt")

for fileName in fileList0:
    os.remove(fileName)

# move files
fileList1 = glob.glob("/users/Wubabear/Desktop/SP02_Mock/*.txt")

for filename in fileList1:
    shutil.copy(filename, '/users/Wubabear/Desktop/SP02_Mock/manually_correct_watson_transcripts_praat')
    
# convert .flac to .wav 
fileList2 = glob.glob('/users/Wubabear/Desktop/SP02_Mock/manually_correct_watson_transcripts_praat/*.flac')

for f in fileList2: 
    file_path = PurePath(f)
    flac_tmp_audio_data = AudioSegment.from_file(file_path, file_path.suffix[1:])
    flac_tmp_audio_data.export(file_path.with_suffix(".wav"))


## Step2: preprocess data in files. 

fileList3 = glob.glob('/users/Wubabear/Desktop/SP02_Mock/manually_correct_watson_transcripts_praat/*.txt')

for file_name in fileList3:
with open(file_name, 'r') as f:
    file_string = f.read()
    
    # remove whitespace, tabs, and line mark (\n). 
    newstring = file_string.split()
    newstring = " ".join(newstring)   
    
    # remove all the digits
    trans_table = str.maketrans("", "", digits)
    newstring = newstring.translate(trans_table)
    
    # remove "%HESITATION"
    newstring = newstring.replace("%HESITATION", "")
    
    # remove dots and replace a dot with a space
    newstring = newstring.replace(". .", ".")
    newstring = newstring.replace(".  .", ".")
    newstring = newstring.replace(".", " ")
    
    # write into each file
    with open(file_name, 'w') as f:
    f.write(newstring)


# Removing whitespace and line marks first converting the entire doc into a string 
# so that you don't need to wrestle with the format of the text file later. 
# If you handle the space after removing certain words and digits you will face 
# different spacing situations between words which migth confuse the replace() 
# and merge some words together. 

### Part III MFA 

## Step1: install MFA. See reference here: https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html

## Step2: run MFA. See documentation and examples here: https://montreal-forced-aligner.readthedocs.io/en/latest/first_steps/example.html#alignment-example

# Jupyter Notebook users: put %%bash in front of the below line before running
# other Python IDE users (e.g., Atom, PyCharm, VSCode), run the below line in your terminal

mfa align ~/Desktop/MFA ~/Documents/MFA/pretrained_models/dictionary/english_us_arpa.dict english_us_arpa ~/Desktop/MFA/aligned

