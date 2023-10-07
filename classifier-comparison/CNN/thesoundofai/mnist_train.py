import torch 
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

BATCH_SIZE=128
if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

EPOCHS = 10
LEARNING_RATE = 0.001

def download_mnist_datasets():
    train_data = datasets.MNIST(
        root='data',
        download=True,
        train=True, 
        transform=ToTensor()
    )
    validation_data = datasets.MNIST(
        root='data',
        download=True,
        train=False, 
        transform=ToTensor()
    )
    return train_data,validation_data

class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() 
        self.dense_layers = nn.Sequential(
            nn.Linear(in_features=28*28,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=10)
        )
        self.softmax = nn.Softmax(dim=1) # Takes all values of the predicted 10 classes and scales them to sum up to 1 (convert to probability)

    def forward(self,input_data):
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)
        return predictions

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
    train_data, _ = download_mnist_datasets()
    print("MNIST dataset downloaded")

    train_data_loader = DataLoader(train_data,batch_size=BATCH_SIZE)

    feed_forward_net = FeedForwardNet().to(DEVICE)
    # Instantiating loss function and optimiser
    loss_function = nn.CrossEntropyLoss()
    optimiser=torch.optim.Adam(feed_forward_net.parameters(),
                            lr=LEARNING_RATE
                                )

    #Train Model
    train(feed_forward_net,train_data_loader, loss_function, optimiser, DEVICE, EPOCHS)

    #Save results
    torch.save(feed_forward_net.state_dict(),"/Users/vedant/Desktop/Programming/learning-pytorch/trained_models/feedforwardnet.pth")

    print("Model trained and stored at /trained_models/feedforwardnet.pth")