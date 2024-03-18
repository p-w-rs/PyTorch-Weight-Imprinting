import os

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image


class ImprintingMachine:
    def __init__(self, label_path=None, relabel_path=None):
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"Using {self.device} device")

        self.model = models.mobilenet_v2(pretrained=True)
        self.model = self.model.to(self.device)
        self.num_pretrained_classes = self.model.classifier[-1].out_features
        self.model.eval()

        self.labels = self.load_labels(label_path)
        self.relabel_path = relabel_path

        self.transform = transforms.Compose(
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

    def run_inference(self, image_path):
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
        _, predicted = torch.max(output, 1)
        return self.labels[predicted.item()]

    def run_imprint(self):
        if self.relabel_path is None:
            raise ValueError("relabel_path must be provided for imprinting.")

        embeddings = []
        label_indices = []

        for label in os.listdir(self.relabel_path):
            label_dir = os.path.join(self.relabel_path, label)
            if label not in self.labels:
                self.labels.append(label)
            label_index = self.labels.index(label)
            label_embeddings = []

            for image_name in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_name)
                image = Image.open(image_path).convert("RGB")
                input_tensor = self.transform(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    embedding = self.model.features(input_tensor)
                    embedding = nn.functional.adaptive_avg_pool2d(embedding, (1, 1))
                    embedding = torch.flatten(embedding, 1)
                    embedding = nn.functional.normalize(embedding, p=2, dim=1)
                    label_embeddings.append(embedding)

            label_embeddings = torch.cat(label_embeddings, dim=0)
            label_embedding = label_embeddings.mean(dim=0, keepdim=True)

            if label_index < self.num_pretrained_classes:
                existing_weight = (
                    self.model.classifier[-1].weight.data[label_index].unsqueeze(0)
                )
                updated_weight = (existing_weight + label_embedding) / 2.0
                embeddings.append(updated_weight)
            else:
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

        self.save_model(os.path.join("test_data", "model_weights.pth"))
        self.save_labels(os.path.join("test_data", "labels.txt"))

    def save_model(self, output_path):
        torch.save(self.model.state_dict(), output_path)

    def save_labels(self, output_path):
        with open(output_path, "w") as f:
            f.write("\n".join(self.labels))
