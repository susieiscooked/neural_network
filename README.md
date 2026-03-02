# CIFAR-10 Image Classifier — PyTorch CNN

A convolutional neural network (CNN) built with PyTorch that classifies images from the CIFAR-10 dataset into 10 categories: plane, car, bird, cat, deer, dog, frog, horse, ship, and truck.

---

## Features

- CNN built from scratch using PyTorch
- Trains on the CIFAR-10 dataset (50,000 training images, 10,000 test images)
- Structured logging with Python's `logging` module
- Model saving and loading with `torch.save` / `torch.load`
- Per-class accuracy reporting
- GPU (CUDA) support with CPU fallback
- Test image visualisation saved to `test_images.png`

---

## Project Structure

```
neural_network/
├── main.py              # Orchestration entrypoint — setup, train, save, evaluate
├── net.py               # CNN model definition, training loop, and inference
├── data_loaders.py      # CIFAR-10 dataset setup and DataLoader construction
├── logger.py            # Structured logging configuration
├── utils.py             # Visualization utilities (imshow)
├── cifar_net.pth        # Saved model weights (generated after training)
├── test_images.png      # Sample test images (generated after training)
├── data/                # CIFAR-10 dataset (auto-downloaded)
└── README.md
```

---

## Requirements

- Python 3.14
- PyTorch
- torchvision
- matplotlib

Install dependencies with:

```bash
uv sync
```

---

## Usage

Run the training and evaluation script:

```bash
uv run main.py
```

This will:
1. Download the CIFAR-10 dataset (first run only)
2. Train the CNN for 2 epochs
3. Save the trained model to `cifar_net.pth`
4. Evaluate accuracy on the test set
5. Print per-class accuracy
6. Save a sample of test images to `test_images.png`

---

## Network Architecture

```
Input (3 x 32 x 32)
  → Conv2d(3, 6, 5) + ReLU + MaxPool
  → Conv2d(6, 16, 5) + ReLU + MaxPool
  → Flatten → Linear(400, 120) + ReLU
  → Linear(120, 84) + ReLU
  → Linear(84, 10)
Output: 10 class scores
```

- **Loss function:** Cross Entropy Loss  
- **Optimizer:** SGD (lr=0.001, momentum=0.9)  
- **Epochs:** 2  
- **Batch size:** 4  

---

## Example Output

```
[1,  2000] loss: 2.136
[1,  4000] loss: 1.762
...
[2, 12000] loss: 1.251
Finished Training

Accuracy of the network on the 10000 test images: 54 %

Accuracy for class: plane is 59.2 %
Accuracy for class: car   is 63.1 %
Accuracy for class: bird  is 41.5 %
...
```

---

## Logging

The project uses Python's built-in `logging` module with `DEBUG` level output, configured via `logger.py`. All output flows through structured logging (no raw `print()` calls), showing timestamps, log level, and function name for every key step in the pipeline.

---

## Notes

- If running on WSL, `matplotlib` is set to use the `Agg` backend (no GUI window). The output image is saved to `test_images.png` and can be viewed in Windows Explorer at `\\wsl$\Ubuntu\home\<your-username>\projects\suja\neural_network\test_images.png`.
- Training from scratch takes approximately 2 minutes on a CUDA-enabled GPU.



