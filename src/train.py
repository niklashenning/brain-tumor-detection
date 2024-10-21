import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
from model import Model
from dataset import BrainTumorDataset


dataset = BrainTumorDataset()

training_size = int(0.80 * len(dataset))
validation_size = len(dataset) - training_size
training_data, validation_data = random_split(dataset, [training_size, validation_size])
training_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=32, shuffle=False)

model = Model()
if torch.cuda.is_available():
    model = model.to(torch.device('cuda'))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
losses = []

for epoch in range(epochs):
    for images, labels in training_dataloader:
        if torch.cuda.is_available():
            images = images.to(torch.device('cuda'))
            labels = labels.to(torch.device('cuda'))

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    losses.append(loss.item())
    print('Epoch [{}/{}], Loss: {:.6f}'.format(epoch + 1, epochs, loss.item()))


model.eval()
correct = 0

with torch.no_grad():
    for images, labels in validation_dataloader:
        if torch.cuda.is_available():
            images = images.to(torch.device('cuda'))
            labels = labels.to(torch.device('cuda'))

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / validation_size
print('Test Accuracy: {:.2f}% ({}/{})'.format(accuracy, correct, validation_size))


plt.get_current_fig_manager().set_window_title('Training')
plt.plot(range(epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
