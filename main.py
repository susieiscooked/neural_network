#!/usr/bin/env python3

# import necessary libraries
import torch
import torchvision
import matplotlib

from utils import imshow
from net import Net, run_inference, train
from logger import setup_logger
from data_loaders import get_images_labels, setup_data_loaders
matplotlib.use('Agg')  # non-interactive backend, no window needed


if __name__ == "__main__":
    logger = setup_logger()

    logger.debug("setup and running")
    if torch.cuda.is_available():
        torch.device("cuda")
        logger.info("Using CUDA")
    else:
        logger.info("CUDA not available")
    
    trainloader, testloader = setup_data_loaders(logger)

    classes =('plane','car','bird','cat','deer','dog','frog', 'horse','ship', 'truck')

    logger.debug("iterate over data loaders")
    images, labels = get_images_labels(testloader)
    imshow(torchvision.utils.make_grid(images))
    logger.info('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    trained_model = train(logger, trainloader)

    PATH = './cifar_net.pth'
    torch.save(trained_model.state_dict(), PATH)

    # Load in our saved model
    trained_model = Net()
    trained_model.load_state_dict(torch.load(PATH, weights_only=True))

    correct, \
    total, \
    correct_pred_per_label, \
    total_pred_per_label = run_inference(classes, testloader, trained_model, logger)

    logger.info(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # print accuracy for each class so we knom which class performed well and which didnot
    for classname, correct_count in correct_pred_per_label.items():
        accuracy = 100 * float(correct_count) / total_pred_per_label[classname]
        logger.info(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
