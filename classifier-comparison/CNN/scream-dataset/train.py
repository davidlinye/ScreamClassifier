from screamdataset import ScreamDataset
import torch 
import torchaudio
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
#from cnn2 import CNNNetwork
from cnn import CNNNetwork
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score
import pandas as pd
import plotly.express as px

TRAIN_ANNOTATIONS_FILE = '/home/vedant/projects/ScreamDetection/resources/dataset/pytorch-dataset-train.csv'
TEST_ANNOTATIONS_FILE = '/home/vedant/projects/ScreamDetection/resources/dataset/pytorch-dataset-test.csv'
VALID_ANNOTATIONS_FILE = '/home/vedant/projects/ScreamDetection/resources/dataset/pytorch-dataset-validation.csv'

TRAIN_AUDIO_DIR = '/home/vedant/projects/ScreamDetection/resources/dataset/blocked_audio/train'
TEST_AUDIO_DIR = '/home/vedant/projects/ScreamDetection/resources/dataset/blocked_audio/test'
VALID_AUDIO_DIR = '/home/vedant/projects/ScreamDetection/resources/dataset/blocked_audio/validation'

BATCH_SIZE = 1024
SAMPLE_RATE=44100
EPOCHS = 5
LEARNING_RATE = 0.001

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

def create_data_loader(train_data,batch_size=None):
    dataloader = DataLoader(train_data,batch_size=batch_size)
    return dataloader

def train_one_epoch(model, train_data_loader,test_data,loss_function,optimiser,device):
    for inputs,targets in train_data_loader:
        model.train()
        inputs,targets = inputs.to(device),targets.to(device)

        # Calculate Loss
        predictions = model(inputs)
        loss = loss_function(predictions,targets)

        # Backpropagate Loss, update weights
        optimiser.zero_grad()
        loss.backward() # Apply backpropagation
        optimiser.step() # Update weights
        #print([prediction.argmax[0] for prediction in predictions])
        class_mapping=[
                    'no_vocals',
                    'midfry',
                    'clean',
                    'highfry',
                    'lowfry',
                    'layered'
                ]
        pred=[]
        for prediction in predictions.detach().cpu():
            p = prediction[0].argmax[0]
            pred.append(p)
        print(pred)
        #EVAL
        # class_mapping=[
        #             'no_vocals',
        #             'midfry',
        #             'clean',
        #             'highfry',
        #             'lowfry',
        #             'layered'
        #         ]
        # model.eval()
        # with torch.no_grad():
        #     predictions=[]
        #     expectation=[]
        #     for i in range(len(test_data)):
        #         inputs,targets=test_data[i]
        #         ip=inputs.unsqueeze(0)
        #         prediction = model(ip)
        #         predicted_index = prediction[0].argmax(0) #Find the predicted class with highest probability
        #         predicted = class_mapping[predicted_index]
        #         expected = class_mapping[targets]
        #         predictions.append(predicted)
        #         expectation.append(expected)

        accuracy = accuracy_score(pred,targets.detach().cpu())
        macro_accuracy = precision_score(pred,targets.detach().cpu(),average='macro')

    print(f"Loss : {loss.item()}")
    print(f"Accuracy: {accuracy}")
    print(f"Macro Accuracy: {macro_accuracy}")
    return loss.item(),accuracy,macro_accuracy

def train(model, train_data_loader,test_data,validation_data,loss_function, optimiser, device, epochs):
    losses=[]
    epoch=[]
    accuracies=[]
    macro_accuracies=[]
    for i in range(epochs):
        epoch.append(i)
        print(f"Epoch {i+1}:")
        loss,accuracy,macro_accuracy = train_one_epoch(model, train_data_loader,test_data, loss_function, optimiser, device)
        losses.append(loss)
        accuracies.append(accuracy)
        macro_accuracies.append(macro_accuracy)
        print("-------------------------------------------------------")
    print("Training done")
    # #Find validation loss
    # model.eval()
    # class_mapping=[
    #                 'no_vocals',
    #                 'midfry',
    #                 'clean',
    #                 'highfry',
    #                 'lowfry',
    #                 'layered'
    #             ]
    # with torch.no_grad():
    #     predictions=[]
    #     expectation=[]
    #     for i in range(len(validation_data)):
    #         inputs,targets=validation_data[i]
    #         inputs = inputs.to(device)
    #         ip = inputs.unsqueeze(0)
    #         prediction = model(ip)
    #         predicted_index = prediction[0].argmax(0) #Find the predicted class with highest probability
    #         predicted = class_mapping[predicted_index]
    #         expected = class_mapping[targets]
    #         predictions.append(predicted)
    #         expectation.append(expected)
    #     accuracy = accuracy_score(predictions,expectation)

    #     macro_accuracy = precision_score(predictions,expectation,average='macro')
    # print(f"Accuracy Score : {accuracy}")
    # print(f"Macro Accuracy : {macro_accuracy}")
    return losses,accuracies,macro_accuracies,epoch


if __name__ == '__main__':
    from sklearn.metrics import confusion_matrix,accuracy_score, precision_score
    #import sys
    #sys.setrecursionlimit(10000)
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

    sd_train = ScreamDataset(TRAIN_ANNOTATIONS_FILE, TRAIN_AUDIO_DIR, mel_spectrogram, DEVICE)
    train_dataloader= create_data_loader(sd_train,BATCH_SIZE)

    sd_test = ScreamDataset(TEST_ANNOTATIONS_FILE, TEST_AUDIO_DIR, mel_spectrogram, DEVICE)
    # test_dataloader= create_data_loader(sd_test,BATCH_SIZE)
    # Loading entire dataset into lists
    test_inputs=[]
    for i in range(len(sd_test)):
        test_inputs.append(sd_test[i])

    sd_valid = ScreamDataset(VALID_ANNOTATIONS_FILE, VALID_AUDIO_DIR, mel_spectrogram, DEVICE)
    # valid_dataloader= create_data_loader(sd_valid,BATCH_SIZE)
    valid_inputs=[]
    for i in range(len(sd_valid)):
        valid_inputs.append(sd_valid[i])

    cnn = CNNNetwork().to(DEVICE)
    # Instantiating loss function and optimiser
    loss_function = nn.CrossEntropyLoss()
    optimiser=torch.optim.Adam(cnn.parameters(),
                            lr=LEARNING_RATE
                                )

    #Train Model (model, train_data_loader,test_data_loader,validation_data_loader,loss_function, optimiser, device, epochs)
    losses,accuracies,macro_accuracies,epoch = train(cnn,train_dataloader, test_inputs, valid_inputs, loss_function, optimiser, DEVICE, EPOCHS)

    #Save results
    torch.save(cnn.state_dict(),"/home/vedant/projects/ScreamDetection/CNN/trained_models/scream_cnn_crossentropy_adam.pth")

    print("Model trained and stored at /CNN/trained_models/scream_cnn_crossentropy_adam.pth")
    # fig = px.line(x=epoch,y=losses)
    # fig.add_scatter(x=epoch, y=accuracies, mode='lines')
    # fig.add_scatter(x=epoch, y=macro_accuracies, mode='lines')
    # # fig.add(px.line(x=epoch,y=accuracies))
    # # fig.add(px.line(x=epoch,y=macro_accuracies))
    # fig.show()
    df=pd.DataFrame()
    df['epoch'] = epoch
    df['accuracy'] = accuracies
    df['macro_accuracy'] = macro_accuracies

    df.to_csv('/home/vedant/projects/ScreamDetection/CNN/trained_models/training_results.csv',header=True,index=False,encoding='utf-8-sig')