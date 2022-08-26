import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch
import cv2
from PIL import ImageOps, Image
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time

load = './CharacterRecognition.pth'
#load = None
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,3)
        #self.fc4 = nn.Linear(40,2)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = self.fc3(x)
        return x
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
net = Net()
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)




transform = transforms.Compose([
    #grayscale
    #transforms.Grayscale(1),
    # resize
    transforms.Resize(32),
    # center-crop
    transforms.CenterCrop(32),
    # to-tensor
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # normalize
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    #transforms.Grayscale(1),
])
trainset = torchvision.datasets.ImageFolder('gameData/', transform = transform) 
#testset = torchvision.datasets.ImageFolder('gameData\\DataTest', transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=True)
testloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=True)
classes = ('back', 'dk', 'game')
def evaluate():
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on test images: {100 * correct // total} %')

if load is None:
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1000 == 0:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    PATH = './CharacterRecognition.pth'
    torch.save(net.state_dict(), PATH)
    print('Finished Training')
    evaluate()
else:
    net.load_state_dict(torch.load(load))

def test():
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(1)))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                for j in range(1)))

def getType(image):
    
    
    image = image.convert('RGB')
    #image = ImageOps.grayscale(image)
    #image.show()
    
    image = transform(image).unsqueeze(0)
    #image = Variable(image, requires_grad=True)
    #mage = image.unsqueeze(0)
    
    with torch.no_grad():
        net.eval()
        
        output = net(image)
        _, prediction = torch.max(output, 1)
        return int(prediction)

#im = Image.open('Capture1.png')#.convert('RGB')
# im = Image.open('Game.png')
# im.show()

#print(getType(im))
#average out time taken to pass 100 images
avg = 0

# if main
if __name__ == '__main__':
    for i in range(100):
        im = Image.open('dk.png').convert('RGB')
        im = transform(im).unsqueeze(0)
        with torch.no_grad():
            net.eval()
            t = time.time()
            output = net(im)
            _, prediction = torch.max(output, 1)
            print(int(prediction))
            avg += time.time() - t

    print(avg/100)
    input("asdf")

def predict(image):
    #conver array to image
   
  
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        net.eval()
        output = net(image)
        _, prediction = torch.max(output, 1)
        return int(prediction)

#divide image into 5 pieces
def divide(image, amount):
    #get width and height of image
    width = image.shape[0]
    height = image.shape[1]

  
    w = width//amount
    h = height//amount
    #turn np array into image
    image = Image.fromarray(image)
    imgs = []
    for i in range(5):
        for j in range(5):
            imgs.append([image.crop((i*w, j*h, (i+1)*w, (j+1)*h)), i, j])
    return imgs, w, h


    