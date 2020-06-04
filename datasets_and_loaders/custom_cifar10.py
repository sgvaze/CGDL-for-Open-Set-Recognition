import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset

from PIL import Image

import numpy as np
from math import floor

class SubSampleLoaderCreator(object):

    """

    Class takes a training set and returns a Pytorch dataloader.
    Data loader returns batches of images/labels only belonging to class specified in selected_classes
    Interface also allows specification of whether loader should be for validation or training, or cover entire set

    """

    def __init__(self, dataset=None, num_subsample_classes=4, batch_size=128, valid=False,
                 num_examples=None):

        self.dataset = dataset
        self.num_subsample_classes = num_subsample_classes
        self.validation = valid

        self.batch_size = batch_size
        self.num_samples = None

        self.train_idxs = None
        self.valid_idxs = None

        self.num_examples = num_examples    # Specify how many images the dataset should contain

    def make_sample_weights_for_class_selection(self, selected_classes):

        # Function returns array of weights (length=number of images). Weight specifies likelihood of dataloader
        # returning the sample. For this purpose, weights are either 0 (image does not belong to selected class)
        # or 1 (image belongs). Also sets some elements to zero to based to allow training or validation loader

        n_images = len(self.dataset)
        first = True
        image_weights = np.zeros((n_images,))

        for class_idx in selected_classes:

            indices = [i for i, x in enumerate(self.dataset.targets) if x == class_idx]

            if self.num_examples is not None:

                num_examples_per_class = floor(self.num_examples/len(selected_classes))

                if first:
                    num_examples_per_class += self.num_examples % len(selected_classes)
                    first = False

                indices = indices[:num_examples_per_class]

            image_weights[indices] = 1

        if self.validation is None:
            return image_weights

        else:

            if self.train_idxs is None:
                self.create_valid_idxs(image_weights=image_weights)

            if not self.validation:

                image_weights[self.valid_idxs] = 0
                return list(image_weights)

            elif self.validation:
                image_weights[self.train_idxs] = 0
                return list(image_weights)


    def get_loader(self, selected_classes, num_workers=4):

        # Create dataloader

        weights = self.make_sample_weights_for_class_selection(selected_classes)
        weights = torch.DoubleTensor(weights)

        self.num_samples = int(torch.sum(weights))

        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, self.num_samples, replacement=False)
        loader = torch.utils.data.DataLoader(self.dataset, shuffle=False,
                                             batch_size=self.batch_size,
                                             sampler=sampler, num_workers=num_workers)

        return loader

    def create_valid_idxs(self, image_weights):

        n_train_images = int(0.8 * image_weights.sum())
        true_locations = np.argwhere(image_weights == 1)
        true_locations = true_locations.reshape((true_locations.size,))

        self.train_idxs = np.random.choice(true_locations, size=n_train_images)

        mask = np.isin(true_locations, self.train_idxs)

        self.valid_idxs = true_locations[~mask]

    def __call__(self, selected_classes, valid=False, dataset=None, num_examples=None):

        # External interface

        # Allow user to change dataset if required
        if dataset is not None:
            self.dataset = dataset

        if isinstance(self.dataset, CustomCIFAR10):
            self.dataset.assign_class_map(selected_classes)

        self.validation = valid
        self.num_examples = num_examples

        return self.get_loader(selected_classes)

class CustomCIFAR10(CIFAR10):

    """
    Class for choosing subsets of CIFAR10 classes to train
    Use with SubSampleLoaderCreator
    """

    def __init__(self, root, train, download, transform):

        CIFAR10.__init__(self, root=root, train=train, download=download, transform=transform)
        self.selected_classes = [5, 6, 7, 8] # Make sure to assign later

        self.class_map_dict = {} # Is assigned later

    def __getitem__(self, index):

        """
                Args:
                    index (int): Index

                Returns:
                    tuple: (image, target) where target is index of the target class.
                """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        target = self.class_map_dict[target]

        return img, target

    def assign_class_map(self, selected_classes):

        self.selected_classes = selected_classes
        self.selected_classes.sort()

        for i, cls in enumerate(self.selected_classes):

            self.class_map_dict[cls] = i