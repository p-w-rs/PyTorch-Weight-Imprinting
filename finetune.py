import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

# Load the pre-trained MobileNetV3 Large model
model = models.mobilenet_v3_large(pretrained=True)

# Modify the model for weight imprinting
# Remove the last classification layer
num_features = model.classifier[-1].in_features
model.classifier = model.classifier[:-1]

# Add L2 normalization layer
model.classifier.add_module("l2_norm", nn.LayerNorm(num_features))

# Add a new linear layer for classification without bias
num_classes = 10  # Number of classes in CIFAR10
model.classifier.add_module("linear", nn.Linear(num_features, num_classes, bias=False))

# Set the last layers to be trainable
for param in model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True

# Move the model to GPU if available
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
model.to(device)

# load CIFAR10 dataset and make resize to be 224 by 224
train_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

trainloader = torch.utils.data.DataLoader(
    CIFAR10(
        root="./test_data",
        train=True,
        download=True,
        transform=train_transform,
    ),
    batch_size=32,
    shuffle=True,
)

testloader = DataLoader(
    CIFAR10(
        root="./test_data",
        train=False,
        download=True,
        transform=train_transform,
    ),
    batch_size=512,
    shuffle=False,
)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

# Fine-tune the last layers
num_epochs = 10
best_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(trainloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Evaluate on the validation set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(testloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader):.4f}, Val Acc: {val_acc:.4f}"
    )

    # Save the best model based on validation accuracy
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model, "test_data/mobilenet_l2norm_v3.pt")

print("Fine-tuning finished. Best model saved.")
