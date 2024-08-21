from typing import Iterable
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import ImageFilter, Image
import random


def otsu_mask(img):
    median = np.median(img)
    _, thresh = cv2.threshold(img, median, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


class OtsuCut(object):

    def __init__(self):
        super().__init__()

    def __process__(self, x):
        if isinstance(x, Image.Image):
            x = np.array(x)
        mask = otsu_mask(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY))
        # Convert to NumPy array if not already

        # Check if the matrix is empty or has no '1's
        if mask.size == 0 or not np.any(mask):
            return Image.fromarray(x)

        # Find the rows and columns where '1' appears
        rows = np.any(mask == 255, axis=1)
        cols = np.any(mask == 255, axis=0)

        # Find the indices of the rows and columns
        min_row, max_row = np.where(rows)[0][[0, -1]]
        min_col, max_col = np.where(cols)[0][[0, -1]]

        # Crop and return the submatrix
        x = x[min_row:max_row+1, min_col:max_col+1]
        img = Image.fromarray(x)
        return img

    def __call__(self, x):
        if isinstance(x, Iterable):
            return [self.__process__(im) for im in x]
        else:
            return self.__process__(x)


class DataTransforms(object):
    def __init__(self, is_train: bool = True, img_size: int = 256, crop_size: int = 224):
        if is_train:
            data_transforms = [
                OtsuCut(),
                transforms.Resize((img_size, img_size)),
                transforms.RandomResizedCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5,  0.5, 0.5))
            ]
        else:
            data_transforms = [
                OtsuCut(),
                transforms.Resize((img_size, img_size)),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]

        self.data_transforms = transforms.Compose(data_transforms)

    def __call__(self, image):
        return self.data_transforms(image)


class DetectionDataTransforms(object):
    def __init__(self, is_train: bool = True, crop_size: int = 224, jitter_strength: float = 1.):
        if is_train:
            self.color_jitter = transforms.ColorJitter(
                0.8 * jitter_strength,
                0.8 * jitter_strength,
                0.8 * jitter_strength,
                0.2 * jitter_strength,
            )

            kernel_size = int(0.1 * 224)
            if kernel_size % 2 == 0:
                kernel_size += 1

            data_transforms = [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        else:
            data_transforms = [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]

        self.data_transforms = transforms.Compose(data_transforms)

    def __call__(self, image):
        return self.data_transforms(image)


class GaussianBlur:
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, p=0.5, min=0.1, max=2.0):

        self.min = min
        self.max = max

        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < self.p:
            sigma = (self.max - self.min) * \
                np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(
                sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


class SimCLRTransform(object):
    def __init__(self, is_train: bool = True, img_size: int = 256, crop_size: int = 224):
        if is_train:
            self.data_transforms = transforms.Compose(
                [
                    OtsuCut(),
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomResizedCrop(size=crop_size, scale=(0.1, 1.0)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([transforms.GaussianBlur((7, 7), [0.1, 2.0])], p=0.4),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ]
            )
        else:
            self.data_transforms = transforms.Compose(
                [
                    OtsuCut(),
                    transforms.Resize((img_size, img_size)),
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ]
            )

    def __call__(self, img):
        return self.data_transforms(img)

class Moco2Transform(object):
    def __init__(self, is_train: bool = True, img_size: int = 256, crop_size: int = 224) -> None:
        if is_train:
            # This setting follows SimCLR
            self.data_transforms = transforms.Compose(
                [
                    OtsuCut(),
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomCrop(crop_size),
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.2, 0.2)], p=0.4),
                    transforms.RandomApply([transforms.GaussianBlur((7, 7), [0.1, 2.0])], p=0.4),
                    transforms.RandomAffine(degrees=10, scale=(0.8,1.1), translate=(0.0625,0.0625)),
                    # output image must be gray scale
                    transforms.RandomGrayscale(p=1.0),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ]
            )
        else:
            self.data_transforms = transforms.Compose(
                [
                    OtsuCut(),
                    transforms.Resize((img_size, img_size)),
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ]
            )

    def __call__(self, img):
        return self.data_transforms(img)


