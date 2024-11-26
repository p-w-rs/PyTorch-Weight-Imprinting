# test.py
import os
import random
import shutil
import time
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from tqdm import tqdm
from ImprintingMachine import ImprintingMachine

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model_path = "test_data/mobilenet_l2norm_v3.pt"
label_path = "test_data/cifar10_labels.txt"
relabel_path = "test_data/relabel"
reinforced_relabel_path = "test_data/reinforced_relabel"

dataset = CIFAR10(root="./test_data", train=False, download=True)
testloader = DataLoader(
    CIFAR10(root="./test_data", train=False, download=True, transform=transform),
    batch_size=512, shuffle=False
)

def reinforce_relabel(machine, relabel_path, reinforced_relabel_path, dataset):
    os.makedirs(reinforced_relabel_path, exist_ok=True)

    for label in os.listdir(relabel_path):
        label_dir = os.path.join(relabel_path, label)
        if os.path.exists(label_dir):
            reinforced_label_dir = os.path.join(reinforced_relabel_path, label)
            shutil.copytree(label_dir, reinforced_label_dir, dirs_exist_ok=True)

    for label in machine.labels:
        label_dir = os.path.join(reinforced_relabel_path, label)
        os.makedirs(label_dir, exist_ok=True)

        cifar_label = dataset.class_to_idx[label]
        cifar_indices = [i for i, (_, l) in enumerate(dataset) if l == cifar_label]
        num_reinforcement = min(len(cifar_indices), 10)
        reinforcement_indices = random.sample(cifar_indices, num_reinforcement)

        for idx in reinforcement_indices:
            image, _ = dataset[idx]
            image_name = f"reinforcement_{idx}.png"
            target_path = os.path.join(label_dir, image_name)
            image.save(target_path)

def test_accuracy(machine, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(machine.device)
            labels = labels.to(machine.device)
            outputs = machine.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def test_imprinted_accuracy(machine, relabel_path):
    correct_imprinted = 0
    total_imprinted = 0
    for label in os.listdir(relabel_path):
        label_dir = os.path.join(relabel_path, label)
        for image_name in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_name)
            inference_result = machine.run_inference(image_path)
            predicted_label = max(inference_result, key=inference_result.get)
            total_imprinted += 1
            if predicted_label == label:
                correct_imprinted += 1
    return correct_imprinted / total_imprinted

def test_imprinting(machine, relabel_path, testloader, reinforce=False):
    accuracy_before = test_accuracy(machine, testloader)
    print(f"Accuracy before imprinting: {accuracy_before:.4f}")

    start_time = time.time()
    machine.relabel_path = relabel_path
    dataset = machine.run_imprint()
    imprinting_time = time.time() - start_time
    print(f"\nImprinting time: {imprinting_time:.2f} seconds")

    accuracy_after = test_accuracy(machine, testloader)
    accuracy_imprinted = test_imprinted_accuracy(machine, relabel_path)
    print(f"Accuracy after imprinting: {accuracy_after:.4f}")
    print(f"Accuracy on imprinted images: {accuracy_imprinted:.4f}")

    start_time = time.time()
    machine.fine_tune_model(dataset, num_epochs=10, batch_size=32, lr=0.001)
    fine_tuning_time = time.time() - start_time
    print(f"\nFine-tuning time: {fine_tuning_time:.2f} seconds")

    accuracy_after = test_accuracy(machine, testloader)
    accuracy_imprinted = test_imprinted_accuracy(machine, relabel_path)
    print(f"Accuracy after fine-tuning: {accuracy_after:.4f}")
    print(f"Accuracy on imprinted images: {accuracy_imprinted:.4f}")

if __name__ == "__main__":
    reinforce_relabel(
        ImprintingMachine(model_path, label_path, None),
        relabel_path,
        reinforced_relabel_path,
        dataset,
    )

    print("Testing imprinting without reinforcement...")
    machine = ImprintingMachine(model_path, label_path, None)
    dataset = test_imprinting(machine, relabel_path, testloader)

    print("\n\n\nTesting imprinting with reinforcement...")
    machine = ImprintingMachine(model_path, label_path, None)
    dataset = test_imprinting(machine, reinforced_relabel_path, testloader)

    machine.save_model_pt("test_data/imprinted_model.pt")
    machine.save_model_onnx("test_data/imprinted_model.onnx")
