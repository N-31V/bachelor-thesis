from torchvision import models
from classificationmodels import *
from preprocessdata import *
import time

num_epoch = 100
num_val = 5
batch_size = 20
plates = PreprocessData('./datasets/plates/')

resNet18forPlates = ClassificationModel(models.resnet18(pretrained=True), 2, 'resNet18forPlates')
start_time = time.time()
resNet18forPlates.train_model(plates, num_epoch, num_val, batch_size)
print("{:.2f} minutes".format((time.time() - start_time)/60))

resNet50forPlates = ClassificationModel(models.resnet50(pretrained=True), 2, 'resNet50forPlates')
start_time = time.time()
resNet50forPlates.train_model(plates, num_epoch, num_val, batch_size)
print("{:.2f} minutes".format((time.time() - start_time)/60))

resNeXt101forPlates = ClassificationModel(models.resnext101_32x8d(pretrained=True), 2, 'resNeXt101forPlates')
start_time = time.time()
resNeXt101forPlates.train_model(plates, num_epoch, num_val, batch_size)
print("{:.2f} minutes".format((time.time() - start_time)/60))

resNet18forPlates = load_model('resNet18forPlates-10-0.7961')
resNet18forPlates.plot_metrics(num_val)
resNet18forPlates.count_metrics_for_class(1)

resNet50forPlates = load_model('resNet50forPlates-10-0.7882')
resNet50forPlates.plot_metrics(num_val)
resNet50forPlates.count_metrics_for_class(1)

resNeXt101forPlates = load_model('resNeXt101forPlates-20-0.8329')
resNeXt101forPlates.plot_metrics(num_val)
resNeXt101forPlates.count_metrics_for_class(1)


start_time = time.time()
test_img_idx, test_predictions = resNet18forPlates.predict(plates, batch_size)
print("{:.3f} seconds".format((time.time() - start_time)/744))

start_time = time.time()
test_img_idx, test_predictions = resNet50forPlates.predict(plates, batch_size)
print("{:.3f} seconds".format((time.time() - start_time)/744))

start_time = time.time()
test_img_idx, test_predictions = resNeXt101forPlates.predict(plates, batch_size)
print("{:.3f} seconds".format((time.time() - start_time)/744))

write_plates_result((test_img_idx, test_predictions), 0.8)




