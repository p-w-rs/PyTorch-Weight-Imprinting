import os
from ImprintingMachine import ImprintingMachine

# model_path = "test_data/mobilenet_v3.onnx"
label_path = "test_data/imagenet_labels.txt"
relabel_path = "test_data/relabel"
images_path = "test_data/images"

machine = ImprintingMachine(label_path, relabel_path)
print("Running inference on images before imprinting:")
for image_dir in os.listdir(images_path):
    label = image_dir
    print("\nClassifying {} images:".format(label))
    for image in os.listdir(os.path.join(images_path, image_dir)):
        print(machine.run_inference(os.path.join(images_path, image_dir, image)))

machine.run_imprint()
print("Running inference on images after imprinting:")
for image_dir in os.listdir(images_path):
    label = image_dir
    print("\nClassifying {} images:".format(label))
    for image in os.listdir(os.path.join(images_path, image_dir)):
        print(machine.run_inference(os.path.join(images_path, image_dir, image)))
