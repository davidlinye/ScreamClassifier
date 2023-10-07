import torch
from cnn import CNNNetwork
#from train import 
import torchaudio
from urbansounddataset import UrbanSoundDataset
from urbansound_train import AUDIO_DIR,ANNOTATIONS_FILE,SAMPLE_RATE,NUM_SAMPLES

class_mapping=[
    "air_conditioner", 
    "car_horn", 
    "children_playing", 
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]

def predict(model,inputs,targets,class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(inputs)
        #Tensor (1,10) -> [[0.1,0.01,....,0.6]]
        predicted_index = predictions[0].argmax(0) #Find the predicted class with highest probability
        predicted = class_mapping[predicted_index]
        expected = class_mapping[targets]
    return predicted, expected

if __name__ == '__main__':
    DEVICE = 'cpu'

    #load back the model
    cnn = CNNNetwork()
    state_dict = torch.load('/home/vedant/projects/ScreamDetection/CNN/trained_models/cnn.pth')
    cnn.load_state_dict(state_dict)

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = 1024,
        hop_length = 512,
        n_mels = 64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE,AUDIO_DIR,mel_spectrogram,SAMPLE_RATE,NUM_SAMPLES,DEVICE)
    
    #get a sample of the urban sound dataset for inference
    
    #[num channels, frequency, time] since its a mel spectrogram,
    #but model expects a 4th dimension of batch_size
    inputs,targets = usd[0][0],usd[0][1] 
    inputs.unsqueeze_(0)#Adding a 4th dimension

    #make an inference
    predicted, expected = predict(cnn,inputs,targets,class_mapping)

    print(f"Predicted = {predicted} and expected = {expected}")