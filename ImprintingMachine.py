import os
import torch
import torch.nn as nn
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
    def __init__(self, model_path, label_path, relabel_path, batch_size=128):
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"Using {self.device} device")

        if model_path is None:
            self.model = models.mobilenet_v3_large(pretrained=True)
        else:
            self.model = torch.load(model_path)

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

        embeddings = []
        label_indices = []
        datasets = []

        for label in os.listdir(self.relabel_path):
            label_dir = os.path.join(self.relabel_path, label)

            if not label in self.labels:
                self.labels.append(label)
            dataset = ImprintingDataset(
                label_dir, self.transform, self.labels.index(label)
            )
            datasets.append(dataset)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

            label_embeddings = []
            for images, _ in dataloader:
                images = images.to(self.device)
                with torch.no_grad():
                    embedding = self.model.features(images)
                    embedding = self.model.avgpool(embedding)
                    embedding = torch.flatten(embedding, 1)
                    for layer in self.model.classifier[:-1]:
                        embedding = layer(embedding)
                    label_embeddings.append(embedding)

            label_embeddings = torch.cat(label_embeddings, dim=0)
            label_index = self.labels.index(label)

            if label_index < self.num_pretrained_classes:
                current_weight = self.model.classifier[-1].weight.data[label_index]
                label_embeddings = torch.cat(
                    [label_embeddings, current_weight.unsqueeze(0)], dim=0
                )

            label_embedding = label_embeddings.mean(dim=0, keepdim=True)
            label_embedding = nn.functional.normalize(label_embedding, p=2, dim=1)
            embeddings.append(label_embedding)
            label_indices.append(label_index)

        embeddings = torch.cat(embeddings, dim=0)

        num_total_classes = len(self.labels)
        new_layer = nn.Linear(
            self.model.classifier[-1].in_features, num_total_classes, bias=False
        )
        new_layer = new_layer.to(self.device)

        new_layer.weight.data[: self.num_pretrained_classes] = (
            self.model.classifier[-1].weight.data[: self.num_pretrained_classes].clone()
        )
        new_layer.weight.data[label_indices] = embeddings
        self.model.classifier[-1] = new_layer

        return ConcatDataset(datasets)

    def fine_tune_model(self, dataset, num_epochs=10, batch_size=32, lr=0.001):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                logits = self.model(images)
                loss = criterion(logits, labels)
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
