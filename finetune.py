import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

# Custom L2 Normalization Layer
class L2Normalize(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)

# Custom Scaled Linear Layer
class ScaledLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.scale = nn.Parameter(torch.Tensor([10.0]))  # Initialize scale parameter as discussed in paper
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # Normalize weights
        normalized_weight = F.normalize(self.weight, p=2, dim=1)
        # Apply scaling factor and compute cosine similarity
        return self.scale * F.linear(x, normalized_weight)

# Load the pre-trained MobileNetV3 Large model
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
# Modify the model for weight imprinting
num_features = model.classifier[0].in_features
model.classifier = nn.Sequential(
    nn.Linear(num_features, 1024),
    L2Normalize(),
    ScaledLinear(1024, 10)
)

# Set the last layers to be trainable
for param in model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Data loading code remains the same...
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trainloader = torch.utils.data.DataLoader(
    CIFAR10(root='./test_data', train=True, download=True, transform=train_transform),
    batch_size=256, shuffle=True
)

testloader = DataLoader(
    CIFAR10(root='./test_data', train=False, download=True, transform=train_transform),
    batch_size=512, shuffle=False
)

# Training code remains the same...
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

# Add a helper function for weight imprinting
def imprint_weights(model, dataloader, num_classes):
    model.eval()
    class_embeddings = {}
    class_counts = {}

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            # Get embeddings (output before the last ScaledLinear layer)
            embeddings = model.features(inputs)
            embeddings = model.avgpool(embeddings)
            embeddings = torch.flatten(embeddings, 1)
            embeddings = model.classifier[:-1](embeddings)

            # Accumulate embeddings for each class
            for emb, label in zip(embeddings, labels):
                label = label.item()
                if label not in class_embeddings:
                    class_embeddings[label] = emb
                    class_counts[label] = 1
                else:
                    class_embeddings[label] += emb
                    class_counts[label] += 1

    # Average embeddings for each class
    for label in class_embeddings:
        class_embeddings[label] /= class_counts[label]
        # Normalize the averaged embedding
        class_embeddings[label] = F.normalize(class_embeddings[label], p=2, dim=0)

    # Stack embeddings to form weight matrix
    weight_matrix = torch.stack([class_embeddings[i] for i in range(num_classes)])

    # Imprint weights
    model.classifier[-1].weight.data = weight_matrix

    return model

# Training loop with periodic imprinting
model = imprint_weights(model, trainloader, num_classes=10)
num_epochs = 10
best_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # Training code remains the same...
    for inputs, labels in tqdm(trainloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Optionally imprint weights periodically
    if (epoch + 1) % 5 == 0:  # Every 5 epochs
        model = imprint_weights(model, trainloader, num_classes=10)

    # Evaluation code remains the same...
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
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader):.4f}, Val Acc: {val_acc:.4f}')

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model, 'test_data/mobilenet_l2norm_v3.pt')

print("Fine-tuning finished. Best model saved.")
