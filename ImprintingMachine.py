
# ImprintingMachine.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset

class ImprintingDataset(Dataset):
    def __init__(self, image_dir, transform, label):
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
        self.transform = transform
        self.label = label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, self.label

class ImprintingMachine:
    class L2Normalize(nn.Module):
        def __init__(self, dim=1):
            super().__init__()
            self.dim = dim

        def forward(self, x):
            return F.normalize(x, p=2, dim=self.dim)

    class ScaledLinear(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.scale = nn.Parameter(torch.Tensor([10.0]))
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.xavier_uniform_(self.weight)

        def forward(self, x):
            normalized_weight = F.normalize(self.weight, p=2, dim=1)
            return self.scale * F.linear(x, normalized_weight)

    def __init__(self, model_path, label_path, relabel_path, batch_size=128):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {self.device} device")

        import sys
        sys.modules['__main__'].L2Normalize = self.L2Normalize
        sys.modules['__main__'].ScaledLinear = self.ScaledLinear

        # Load model with custom pickle mapping
        self.model = torch.load(
            model_path,
            map_location=self.device
        )
        self.model = self.model.to(self.device)
        self.num_pretrained_classes = self.model.classifier[-1].out_features
        self.model.eval()

        self.labels = self.load_labels(label_path)
        self.relabel_path = relabel_path
        self.batch_size = batch_size
        self.transform = self.get_transform()

    def get_transform(self):
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def load_labels(self, label_path):
        with open(label_path, "r") as f:
            labels = [line.strip() for line in f.readlines()]
        return labels

    def run_inference(self, image_or_path, top_k=1):
        self.model.eval()

        if isinstance(image_or_path, str):
            image = Image.open(image_or_path).convert("RGB")
        else:
            image = image_or_path.convert("RGB")

        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = nn.functional.softmax(output, dim=1)

        top_k_prob, top_k_indices = torch.topk(probabilities, top_k, dim=1)
        top_k_prob = top_k_prob.squeeze().tolist()
        top_k_indices = top_k_indices.squeeze().tolist()

        if isinstance(top_k_indices, int):
            top_k_indices = [top_k_indices]
            top_k_prob = [top_k_prob]

        top_k_labels = [self.labels[idx] for idx in top_k_indices]

        return dict(zip(top_k_labels, top_k_prob))

    def run_imprint(self):
        self.model.eval()
        if self.relabel_path is None:
            raise ValueError("relabel_path must be provided for imprinting.")

        # Initialize with existing weights
        class_embeddings = {
            label: weight.clone() for label, weight in
            zip(self.labels, self.model.classifier[-1].weight.data)
        }
        class_counts = {label: 1 for label in self.labels}

        # Process each label directory
        label_dirs = [d for d in os.listdir(self.relabel_path)
                     if os.path.isdir(os.path.join(self.relabel_path, d))]
        datasets = []
        for label_dir in label_dirs:
            if label_dir not in self.labels:
                self.labels.append(label_dir)

            dataset = ImprintingDataset(
                os.path.join(self.relabel_path, label_dir),
                self.transform, label_dir
            )
            datasets.append(dataset)
        combined_dataset = ConcatDataset(datasets)
        dataloader = DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=False)

        # Collect embeddings
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                embeddings = self.model.features(inputs)
                embeddings = self.model.avgpool(embeddings)
                embeddings = torch.flatten(embeddings, 1)
                embeddings = self.model.classifier[:-1](embeddings)

                # Accumulate embeddings
                for emb, label in zip(embeddings, labels):
                    if label not in class_embeddings:
                        class_embeddings[label] = emb
                        class_counts[label] = 1
                    else:
                        class_embeddings[label] += emb
                        class_counts[label] += 1

        # Create new weight matrix
        weight_matrix = []
        for label in self.labels:
            avg_embedding = class_embeddings[label] / class_counts[label]
            normalized_embedding = F.normalize(avg_embedding, p=2, dim=0)
            weight_matrix.append(normalized_embedding)

        # Update model weights
        self.model.classifier[-1].weight.data = torch.stack(weight_matrix)
        self.model.classifier[-1].out_features = len(self.labels)
        self.num_pretrained_classes = len(self.labels)

        return combined_dataset

    def fine_tune_model(self, dataset, num_epochs=10, batch_size=32, lr=0.001):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            for images, labels in dataloader:
                images = images.to(self.device)
                label_indices = torch.tensor([self.labels.index(label) for label in labels],
                                                      device=self.device)
                optimizer.zero_grad()
                logits = self.model(images)
                loss = criterion(logits, label_indices)
                loss.backward()
                optimizer.step()

    def save_model_pt(self, output_path):
        torch.save(self.model, output_path)
        self.save_labels(os.path.join("test_data", "labels.txt"))

    def save_model_onnx(self, output_path):
        input = torch.randn(1, 3, 224, 224).to(self.device)
        torch.onnx.export(self.model, input, output_path, verbose=True)
        self.save_labels(os.path.join("test_data", "labels.txt"))

    def save_labels(self, output_path):
        with open(output_path, "w") as f:
            f.write("\n".join(self.labels))
