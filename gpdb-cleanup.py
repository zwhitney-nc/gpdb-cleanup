# -*- coding: utf-8 -*-
"""
***setup***

make sure this is in your .bashrc or equilavent:
export GOOGLE_APPLICATION_CREDENTIALS="gpdb-cleanup-key.json"
replace "gpdb-cleanup-key.json" with path to your credentials

make sure gcloud cli is installed:
https://cloud.google.com/sdk/docs/install-sdk

make sure FFMPEG CLI and nscd are installed:
(Amazon Linux 2023)
FFMPEG CLI: https://johnvansickle.com/ffmpeg/
sudo dnf install nscd

(Ubuntu)
sudo apt install ffmpeg


set up torch and CUDA:
(see here for more info: https://medium.com/@awcalibr/running-whisper-speech-to-text-recognition-in-aws-4f511e3cb4cb)

pip3 install torch torchvision torchaudio
sudo apt install nvidia-driver-525
sudo reboot (will need to wait a few minutes before sshing back in)

nvidia-smi (to check installation succeeded)

sudo apt install nvidia-cuda-toolkit
(can verify with "nvcc --version")

in Python, check sucessful installation with (should return True):
import torch
torch.cuda.is_available()


Below terminal commands are currently included in this script, but you
may need to adjust them for your system
make sure you're in the gpdb-cleanup folder

log into gcloud (in terminal):
gcloud auth activate-service-account gpdb-cleanup@gpdb-cleanup.iam.gserviceaccount.com --key-file certs/gpdb-cleanup-key.json

(Amazon Linux 2023 only)
start nscd (in terminal):
sudo service nscd start 


"""
#TODO: rewrite to work in smaller chunks and save as you go (append csv)


### import packages
from numpy import argmin, isnan, any, all, intersect1d, arange, random, array, where, logical_not, isnan, concatenate, append
from collections import Counter
from itertools import compress, islice
import pandas as pd

import whisper
from textdistance import mra, editex, hamming
import stopit
from alphabet_detector import AlphabetDetector
from collections import deque
import numpy as np
from google.oauth2 import service_account
import six
from google.cloud import translate_v2 as translate

import timeit

import os


### user variables
#check these before running
breakpoint()

input_fn = "db-exports/GPDB-cleanup-prepped.csv"
output_fn = 'db-exports/gpdb_cleanup_OUTPUT.csv'
key_fn = "/home/ubuntu/gpdb-cleanup/certs/gpdb-cleanup-key.json"
git_commit_hash = 'e536cc119628b22a82f8cc07a0730584bf47b6af'

df = pd.read_csv(input_fn)
df = df[:300].copy()
start = 200
step = 100

#currently just hard-coding these, but you can update if needed
os.system(f'gcloud auth activate-service-account gpdb-cleanup@gpdb-cleanup.iam.gserviceaccount.com --key-file {key_fn}')
#os.system('sudo service nscd start')


### read in data and reformat
#load in csv to check and print the first few rows to check

"""df = df.iloc[start:].copy() 
df.reset_index(drop=True, inplace=True)"""
#print(df.shape)

def format_data(df):
  """helper method to pull relevant columns into dictionary and preprocess

  returned dicitonary will be of the form {"name INDEX", (recording url, date recorded)}
  """
  
  # save the relevant columns as separate variables
  names = df['name_gpdb']
  recs = df['gpdb_url']
  date = df["created_at"]

  # pull saved column data into a dict of the form {"name INDEX", (recording url, date recorded)}
  data = {}
  for ind, name in enumerate(names):
    rec = recs[ind]
    dt = date[ind]
    if type(name)==float: #nan name
      name = "InvalidName "+str(ind)
      data[name] = (rec, dt)
      # data[name] = (rec)
      continue
    if name in data:
      name = name + str(ind)
    if type(rec)==float or rec.isnumeric(): #nan or numerical rec link
      rec = "InvalidRec "+str(ind)
      data[name] = (rec, dt)
      # data[name] = (rec)
      continue
    if name.isnumeric():
      name = "InvalidName "+str(ind)
      data[name] = (rec, dt)
      # data[name] = (rec)
      continue
    if not rec.startswith("https://production-processed-recordings"):
      name = "IncompatibleRec "+str(ind)
    data[name] = (rec, dt)
    # data[name] = (rec)
  #print(dict(islice(data.items(),100)))

  return data


### whisper and Google API setup

model = whisper.load_model("small", device="cuda")

ad = AlphabetDetector()

credentials = service_account.Credentials.from_service_account_file(
    key_fn)

# make sure you've logged into gcloud by now!

def pre_transliterate(word):
  translate_client = translate.Client()
  if isinstance(word, six.binary_type):
      text = text.decode("utf-8")
  print("Google Translating: ", word)
  result = translate_client.translate(word)
  return result["translatedText"]

### primary data processing

def process_names(df: pd.DataFrame,
                  data: dict,
                  ):
  """helper method for the primary data processing work"""
                  
  namekeys = data.keys()
  namekeys = np.array(list(namekeys))

  sil = deque() #for each recording in data, contains whether the recording is silent or not
  targets = deque() #for each recording in data, contains whether the recording matches the target name
  confs = deque() #to contain confidence of target predictions

  """import timeit
  start = timeit.default_timer()"""

  for name in namekeys:
    print(name)
    #rec = data[name]
    rec = data[name][0]
    if rec.startswith("Inval"):
      sil.append("N/A")
      targets.append("N/A")
      confs.append("N/A")
      continue
    elif name.startswith("Incomp"): # incompatible recording - youtube, twitter, etc
      sil.append("N/A")
      targets.append("N/A")
      confs.append("N/A")
      continue
    try:
      audio = whisper.load_audio(rec)
    except RuntimeError:
      sil.append("yes")
      targets.append("no")
      confs.append(100)
      continue
    with stopit.ThreadingTimeout(12) as context_manager1:
      # try:
      first = whisper.transcribe(model, audio)
      # except AssertionError:
      #   continue
    if context_manager1.state == context_manager1.EXECUTED:
      transcription = first["text"]
      empty = ["", " "]
      if transcription in empty:
        sil.append("yes")
        targets.append("no")
        confs.append(100)
        continue
      else:
        sil.append("no")
        transcription = transcription.strip(" ")
        if not ad.is_latin(transcription[-1]):
          transcription = pre_transliterate(transcription)
        names = transcription.split(" ")
        sim = {n:editex.similarity(name, n) for n in names}
        simsort = sorted(sim, key=sim.get)
        tgt = simsort[-1]
        finaldist = editex.distance(name, tgt.lower())
        if finaldist <= 5:
          # print(finaldist)
          targets.append("yes")
          confs.append(round(finaldist*16.67))
          continue
        transcription = [t.strip(",.?") for t in names if t.lower() != name]
        catsim = mra.similarity(name, "".join(transcription))
        if catsim > 2:
          targets.append("yes")
          confs.append(round(catsim*33.33))
        else:
          targets.append("no")
          confs.append(100)
    elif context_manager1.state == context_manager1.TIMED_OUT: # Did code timeout?
        sil.append("yes")
        targets.append("no")
        confs.append(100)
        continue

  """end = timeit.default_timer()
  timeElapsed = end - start

  print(f'took {timeElapsed} seconds for {len(namekeys)} names, {len(namekeys)/timeElapsed} names per second')"""

  # print(list(zip(targets, namekeys[:200]))[:len(sil)], sil, confs)
  # print(list(zip(sil, namekeys))[:len(sil)], targets, confs)

  #package up results and write to new CSV

  # print(len(is_sil.values), len(df))
  #dfnew = df.loc[df["created_at"].str.startswith("2020")][:200]
  dfnew = df.copy()
  index_values = dfnew.index
  dfnew["is_silent"] = pd.Series(np.array(sil), index=index_values)
  dfnew["is_target"] = pd.Series(np.array(targets), index=index_values)
  dfnew["target_confidence_%"] = pd.Series(np.array(confs), index=index_values)

  dfnew.to_csv(output_fn, header=False, index = False, mode='a') #maybe move this out of the method?


### loop for working through dataset

totalStart = timeit.default_timer()

i = start
totalNames = 0
while i < df.shape[0]:
  loopStart = timeit.default_timer()

  end = i + step
  if end > df.shape[0]:
    end = (df.shape[0] - 1)

  df_chunk = df.iloc[i:end].copy()
  df_chunk.reset_index(drop=True, inplace=True)

  data = format_data(df_chunk)

  process_names(df_chunk, data)

  i = end

  totalNames += df_chunk.shape[0]
  loopEnd = timeit.default_timer()
  loopTimeElapsed = loopEnd - loopStart
  totalTimeElapsed = loopEnd - totalStart
  
  print(f'this chunk took {loopTimeElapsed} seconds for {df_chunk.shape[0]} names, {loopTimeElapsed/df_chunk.shape[0]} seconds per name')
  print(f'so far, it has taken {totalTimeElapsed} seconds for {totalNames} names in total, {totalTimeElapsed/totalNames} seconds per name')


totalEnd = timeit.default_timer()
totalTimeElapsed = totalEnd - totalStart

print(f'in total, it took {totalTimeElapsed} seconds for {totalNames} names, {totalTimeElapsed/totalNames} seconds per name')




