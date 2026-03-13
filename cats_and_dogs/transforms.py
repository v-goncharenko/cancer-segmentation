from torchvision import transforms

from cats_and_dogs.constants import IMAGE_MEAN, IMAGE_STD, SIZE_H, SIZE_W


def default_transform():
    return transforms.Compose(
        [
            transforms.Resize((SIZE_H, SIZE_W)),  # scaling images to fixed size
            transforms.ToTensor(),  # converting to tensors, transposing channel dimension
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),  # normalize image data per-channel
        ]
    )
