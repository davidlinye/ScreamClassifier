# Scream Classifier tool

Given a song as input, generates a graph classifying over time whether singing, screaming or instrumentals are present at that given time.

## vocal_classification.ipynb source

We used the following github repository for our notebook file: https://github.com/VedantKalbag/ScreamDetection

To simulate the results of our vocal_classification.ipynb, we provide instructions that can be applied to the above mentioned repository.

## Used dataset

The used dataset consists of three songs from the above mentioned repository + our own annotated dataset of twenty new songs.

Our dataset can be downloaded from https://drive.google.com/drive/folders/1oVt896oSAvDJbEXi9pXESOaZs2_EqYZa?usp=drive_link.

If the above link doesn't work, the songs can be manually downloaded from the YouTube playlist: https://www.youtube.com/playlist?list=PLrtpkXmnxmmeP07Y5SY0o9fc8zy7foBwI

When running the code, the dataset of audio files should be present in the folder ./resources/dataset/Audio

## Instructions for simulating vocal_classification.ipynb with the Scream Detection repo

Instructions for running (not training!) the model on music files

data files needed:
    resources/dataset/lookup.csv
    resources/dataset/final/Annotations/annotations.csv
    resources/dataset/Audio/vocal_only/*_vocal_only.wav
    CNN/trained_models/fcnn/fcnn_layer1-256_batch-1024_epochs-500_lr-5e-05 (entire folder)

code files needed:
    classifier-comparison/vocal-only-classification.ipynb
    preprocessing/vggish.ipynb
    FINAL/fcnn.ipynb

1. Load the vocal_only .wav files from Google Drive
2. Create/find a resources/dataset/lookup.csv and resources/dataset/final/Annotations/annotations.csv file containing only the corresponding songs
3. Run the 'Labelling the blocks based on ground truth' section in classifier-comparison/vocal-only-classification.ipynb
    - Pre: resources/dataset/final/Annotations/annotations.csv & resources/dataset/Audio/vocal_only/*_vocal_only.wav files
    - Post: resources/working_data/vocal_only_data.npy
4. Run the 'Vocal only VGGISH' section of preprocessing/vggish.ipynb
    - Pre: resources/working_data/vocal_only_data.npy
    - Post: resources/working_data/vocal_only_data_with_vggish.npy
    - Warning: if the torch imports crashes, run 'pip install typing-extensions --upgrade'
5. Run the '13 delta_mfccs, ZCR, Spectral Crest, Spectral Centroid' section in classifier-comparison/vocal-only-classification.ipynb
    - Pre: resources/working_data/vocal_only_data_with_vggish.npy
    - Post: resources/resources/working_data/vocal_only_features.npy
6. Run all blocks in FINAL/fcnn.ipynb
    - Pre: resources/resources/working_data/vocal_only_features.npy & resources/dataset/lookup.csv & CNN/trained_models/fcnn/fcnn_layer1-256_batch-1024_epochs-500_lr-5e-05 (entire folder)
    - Post: predicted results for the songs, a confusion matrix and some accuracy scores

## getClassStats.py & getStats.py

Some independent statistic files. Not required to run vocal_classification.ipynb