from torch.utils.data import Dataset
import torch
import pandas as pd
import torchaudio
import os

class ScreamDataset(Dataset):
    def __init__(self,annotation_file, audio_dir,transformation,device):
        self.annotations = pd.read_csv(annotation_file,header=0)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.num_samples = 44100

    def _get_audio_sample_path(self, index):
        filename = f"{self.annotations.iloc[index, 0]}.wav"
        path = os.path.join(self.audio_dir,filename)
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 5]

    def _right_pad_if_necessary(self,signal):
        length_of_signal = signal.shape[1]
        if length_of_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_of_signal
            last_dim_padding = (0,num_missing_samples) # (left_pad,right_pad)
            signal = torch.nn.functional.pad(signal,last_dim_padding)
        return signal

    def __len__(self):
        return len(self.annotations)
 
    def __getitem__(self,index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device) #register signal to device
        signal = self._right_pad_if_necessary(signal)
        #Transformations
        signal = self.transformation(signal)
        return signal, label
 


if __name__ == "__main__":
    TRAIN_ANNOTATIONS_FILE = '/home/vedant/projects/ScreamDetection/resources/dataset/pytorch-dataset-train.csv'
    AUDIO_DIR = '/home/vedant/projects/ScreamDetection/resources/dataset/blocked_audio/train'
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate = 44100,
            n_fft = 1024,
            hop_length = 512,
            n_mels = 64
        )
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'

    sd_train = ScreamDataset(TRAIN_ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, DEVICE)
    sd_test = ScreamDataset(TRAIN_ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, DEVICE)
    sd_valid = ScreamDataset(TRAIN_ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, DEVICE)
    data,label = sd_train[0]
    print(label)
    print(data.shape)
    