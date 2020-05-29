from torchvision import models
from classificationmodels import *
from preprocessdata import *
import time

num_epoch = 3
num_val = 3
batch_size = 20
dogs = PreprocessData('./datasets/dogs/')

resNet18forDogs = ClassificationModel(models.resnet18(pretrained=True), 120)
start_time = time.time()
resNet18forDogs.train_model(dogs, num_epoch, num_val, batch_size)
print("{:.2f} minutes".format((time.time() - start_time)/60))
resNet18forDogs.save_model('resNet18forDogs')

resNet50forDogs = ClassificationModel(models.resnet50(pretrained=True), 120)
start_time = time.time()
resNet50forDogs.train_model(dogs, num_epoch, num_val, batch_size)
print("{:.2f} minutes".format((time.time() - start_time)/60))
resNet50forDogs.save_model('resNet50forDogs')

resNeXt101forDogs = ClassificationModel(models.resnext101_32x8d(pretrained=True), 120)
start_time = time.time()
resNeXt101forDogs.train_model(dogs, num_epoch, num_val, batch_size)
print("{:.2f} minutes".format((time.time() - start_time)/60))
resNeXt101forDogs.save_model('resNeXt101forDogs')

resNet18forDogs = load_model('resNet18forDogs')
resNet18forDogs.plot_metrics(num_val)
resNet18forDogs.count_metrics_for_model()

resNet50forDogs = load_model('resNet50forDogs')
resNet50forDogs.plot_metrics(num_val)
resNet50forDogs.count_metrics_for_model()

resNeXt101forDogs = load_model('resNeXt101forDogs')
resNeXt101forDogs.plot_metrics(num_val)
resNeXt101forDogs.count_metrics_for_model()