#training and testing
import torch
from datetime import datetime
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from utils import validate, show_confMat
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os
from sklearn.model_selection import train_test_split
import numpy as np
import random
from model import (
    SwinTransformer,
    MobileNetV2WithAttention,
)

classes_name = ['AF', 'N']
seed = 520
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


result_dir = 'Result'
now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
log_dir = os.path.join(result_dir, time_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir=log_dir)

# Data pre-processing
transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load data sets
img_dir = r""
dataset = datasets.ImageFolder(img_dir, transform=transform)

# Get an index of AF and N categories
class_indices = {'AF': [], 'N': []}
for idx, (_, label) in enumerate(dataset):
    class_name = dataset.classes[label]
    class_indices[class_name].append(idx)


num_AF = len(class_indices['AF'])
num_N = len(class_indices['N'])
total_samples = num_AF + num_N
af_ratio = num_AF / total_samples
n_ratio = num_N / total_samples

print(f"AF samples: {num_AF}, N samples: {num_N}, AF ratio: {af_ratio:.2%}, N ratio: {n_ratio:.2%}")


valid_size = 0.2
test_size = 0.1
train_indices = []
valid_indices = []
test_indices = []


for class_name, indices in class_indices.items():
    train_idx, temp_idx = train_test_split(indices, test_size=(valid_size + test_size), random_state=seed)
    valid_idx, test_idx = train_test_split(temp_idx, test_size=test_size / (valid_size + test_size), random_state=seed)

    train_indices.extend(train_idx)
    valid_indices.extend(valid_idx)
    test_indices.extend(test_idx)


random.shuffle(train_indices)
random.shuffle(valid_indices)
random.shuffle(test_indices)


train_data = Subset(dataset, train_indices)
valid_data = Subset(dataset, valid_indices)
test_data = Subset(dataset, test_indices)


train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


print(f"Train set size: {len(train_data)} (AF: {af_ratio:.2%}, N: {n_ratio:.2%})")
print(f"Validation set size: {len(valid_data)}")
print(f"Test set size: {len(test_data)}")

# Initialize the model
model = MobileNetV2WithAttention()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=5)


best_valid_loss = float('inf')
best_epoch = -1
best_model_path = os.path.join(log_dir, 'best_model.pth')


train_losses = []
valid_losses = []
train_accs = []
valid_accs = []

epochs = 30
for epoch in range(epochs):
    model.train()
    loss_sigma = 0.0
    correct = 0.0
    total = 0.0
    scheduler.step()

    for i, (data, label) in enumerate(train_loader):
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, dim=1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        loss_sigma += loss.item()

        if i % 10 == 9:
            loss_avg = loss_sigma / 10
            loss_sigma = 0.0
            current_acc = correct / total
            print(f"Training: Epoch[{epoch + 1}/{epochs}] Iteration[{i + 1}/{len(train_loader)}] Loss: {loss_avg:.8f} Acc: {current_acc:.4%}")
            writer.add_scalars('Loss_group', {'train_loss': loss_avg}, epoch)
            writer.add_scalar('learning rate', scheduler.get_last_lr()[0], epoch)
            writer.add_scalars('Accuracy_group', {'train_acc': current_acc}, epoch)


    train_losses.append(loss_avg)
    train_accs.append(correct / total)

    # Validation steps
    model.eval()
    loss_sigma = 0.0
    correct_valid = 0
    total_valid = 0

    with torch.no_grad():
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)
            outputs = model(data)
            loss = criterion(outputs, label)
            loss_sigma += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_valid += label.size(0)
            correct_valid += (predicted == label).sum().item()

    avg_valid_loss = loss_sigma / len(valid_loader)
    valid_acc = correct_valid / total_valid
    valid_losses.append(avg_valid_loss)
    valid_accs.append(valid_acc)
    writer.add_scalars('Loss_group', {'valid_loss': avg_valid_loss}, epoch)
    writer.add_scalars('Accuracy_group', {'valid_acc': valid_acc}, epoch)

    print(f'Epoch [{epoch + 1}/{epochs}], Valid Loss: {avg_valid_loss:.4f}, Valid Acc: {valid_acc:.2%}')

    if avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), best_model_path)
        print(f'Best model saved at epoch {best_epoch} with validation loss {best_valid_loss:.4f}')

print('Finished Training')


model.load_state_dict(torch.load(best_model_path))
model = model.to(device)


conf_mat_train, train_acc = validate(model.cpu(), train_loader, 'train', classes_name)
conf_mat_valid, valid_acc = validate(model.cpu(), test_loader, 'test', classes_name)


show_confMat(conf_mat_train, classes_name, 'train', log_dir)
show_confMat(conf_mat_valid, classes_name, 'test', log_dir)


plt.figure()
plt.plot(range(epochs), train_losses, label='Train Loss')
plt.plot(range(epochs), valid_losses, label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(log_dir, 'loss_curve.png'))


plt.figure()
plt.plot(range(epochs), train_accs, label='Train Accuracy')
plt.plot(range(epochs), valid_accs, label='Valid Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(log_dir, 'accuracy_curve.png'))
