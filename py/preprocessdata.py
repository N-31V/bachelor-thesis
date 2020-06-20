import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ImageFolderWithNames(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithNames, self).__getitem__(index)
        path = self.imgs[index][0]
        return original_tuple[0], path[path.rfind('/') + 1:path.rfind('.')]


class ImageFolderForDogs(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderForDogs, self).__getitem__(index)
        if original_tuple[1] < 100:
            target = original_tuple[1]+151
        elif original_tuple[1] < 117:
            target = original_tuple[1] + 152
        else:
            target = original_tuple[1] + 156
        return original_tuple[0], target


def write_plates_result(data, p):
    submission_df = pd.DataFrame.from_dict({'id': data[0], 'label': data[1]})
    submission_df['label'] = submission_df['label'].map(lambda pred: 'dirty' if pred >= p else 'cleaned')
    submission_df.set_index('id', inplace=True)
    submission_df.to_csv('submission.csv')


class PreprocessData:
    def __init__(self, datasetdir):
        self.img_size = 224
        # self.class_names = class_names
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.datasetdir = datasetdir
        # self.test_dir = datasetdir + 'test/unknown/'

        self.train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
        ])

        self.test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
        ])

        self.train_dataset = torchvision.datasets.ImageFolder(self.datasetdir+'train', self.train_transforms)
        self.val_dataset = torchvision.datasets.ImageFolder(self.datasetdir + 'val', self.test_transforms)
        self.test_dataset = ImageFolderWithNames(datasetdir + 'test', self.test_transforms)
        # self.test_dataloader = torch.utils.data.DataLoader(
        #     self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def train_with_random_resize(self):
        self.train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ColorJitter(),
            # torchvision.transforms.RandomGrayscale(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
        ])
        self.train_dataset = torchvision.datasets.ImageFolder(self.datasetdir+'train', self.train_transforms)

    def show_input(self, input_tensor, title):
        image = input_tensor.permute(1, 2, 0).numpy()
        image = self.std * image + self.mean
        plt.imshow(image.clip(0, 1))
        plt.title(title)
        plt.show()
        plt.pause(0.001)
