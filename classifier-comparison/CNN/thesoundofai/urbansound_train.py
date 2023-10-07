import torch 
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from urbansounddataset import UrbanSoundDataset
from cnn import CNNNetwork

ANNOTATIONS_FILE = '/home/vedant/projects/ScreamDetection/CNN/data/UrbanSound8K/metadata/UrbanSound8K.csv'
AUDIO_DIR = '/home/vedant/projects/ScreamDetection/CNN/data/UrbanSound8K/audio' 
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
BATCH_SIZE=1024
EPOCHS = 10#30
LEARNING_RATE = 0.001

def create_data_loader(train_data,batch_size):
    train_dataloader = DataLoader(train_data,batch_size=batch_size)
    return train_dataloader

def train_one_epoch(model, data_loader,loss_function,optimiser,device):
    for inputs,targets in data_loader:
        inputs,targets = inputs.to(device),targets.to(device)

        # Calculate Loss
        predictions = model(inputs)
        loss = loss_function(predictions,targets)

        # Backpropagate Loss, update weights
        optimiser.zero_grad()
        loss.backward() # Apply backpropagation
        optimiser.step() # Update weights
    print(f"Loss : {loss.item()}")

def train(model, data_loader,loss_function, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}:")
        train_one_epoch(model, data_loader, loss_function, optimiser, device)
        print("-------------------------------------------------------")
    print("Training done")



if __name__ == '__main__':
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'

    print(f"Using device: {DEVICE}")

    

    #instantiating dataset object and transform
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = 1024,
        hop_length = 512,
        n_mels = 64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE,AUDIO_DIR,mel_spectrogram,SAMPLE_RATE,NUM_SAMPLES,DEVICE)
    train_dataloader = create_data_loader(usd,BATCH_SIZE)

    cnn = CNNNetwork().to(DEVICE)
    # Instantiating loss function and optimiser
    loss_function = nn.CrossEntropyLoss()
    optimiser=torch.optim.Adam(cnn.parameters(),
                            lr=LEARNING_RATE
                                )

    #Train Model
    train(cnn,train_dataloader, loss_function, optimiser, DEVICE, EPOCHS)

    #Save results
    torch.save(cnn.state_dict(),"/home/vedant/projects/ScreamDetection/CNN/trained_models/cnn.pth")

    print("Model trained and stored at /trained_models/cnn.pth")