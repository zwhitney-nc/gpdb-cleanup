#! /bin/bash

pip install google-colab

pip install git+https://github.com/openai/whisper.git

# pwd

# this won't work in order
# ls drive/Shareddrives/Data\ Department/Data\ Assets/Phonetics

pip install pandas

pip install textdistance
pip install stopit
pip install alphabet_detector

cp /content/drive/Shareddrives/Data\ Department/Data\ Assets/Phonetics/gpdb-cleanup-key.json /content/.config/
export GOOGLE_APPLICATION_CREDENTIALS="/content/.config/gpdb-cleanup-key.json" #need to make sure you copy this from Data Assets/Phonetics

pip install google-api-core
pip install google-cloud-translate
pip install idna

gcloud auth application-default login --impersonate-service-account gpdb-cleanup@gpdb-cleanup.iam.gserviceaccount.com
gcloud config set project gpdb-cleanup


