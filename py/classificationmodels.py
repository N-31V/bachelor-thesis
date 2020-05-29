import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import pickle


def load_model(name):
    with open('models/' + name + '.pickle', 'rb') as f:
        model = pickle.load(f)
    return model


class ClassificationModel:

    def __init__(self, model, num_class, name):
        self.model = model
        self.name = name
        self.num_class = num_class
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = torch.nn.Linear(model.fc.in_features, num_class)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device: ", self.device)
        self.model = model.to(self.device)

        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.5)

        self.train_loss_hist = [1.]
        self.train_acc_hist = [0.]
        self.val_loss_hist = [1.]
        self.val_acc_hist = [0.]
        self.epochs = 0
        self.val_predictions = []
        self.val_targets = []
        self.best_val_acc = 0

    def train_model(self, data, num_epochs, n_val, batch_size):
        for epoch in range(1, num_epochs + 1):
            if epoch == 10:
                data.train_with_random_resize()
            self.epochs += 1
            print('Epoch {}/{}:'.format(epoch, num_epochs), flush=True)
            dataloader = torch.utils.data.DataLoader(
                data.train_dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size, drop_last=True)
            self.model.train()
            running_loss = 0.
            running_acc = 0.
            for inputs, labels in tqdm(dataloader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    preds = self.model(inputs)
                    loss_value = self.loss(preds, labels)
                    preds_class = preds.argmax(dim=1)
                    loss_value.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                running_loss += loss_value.item()
                running_acc += (preds_class == labels.data).float().mean()
            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_acc / len(dataloader)
            self.train_loss_hist.append(epoch_loss)
            self.train_acc_hist.append(epoch_acc)
            print('train loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc), flush=True)

            if epoch % n_val == 0:
                self.val_model(data, batch_size)

    def val_model(self, data, batch_size):
        val_predictions = []
        val_targets = []
        dataloader = torch.utils.data.DataLoader(data.val_dataset, batch_size=batch_size, shuffle=False,
                                                 num_workers=batch_size)
        self.model.eval()  # Set model to evaluate mode
        running_loss = 0.
        running_acc = 0.
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            with torch.set_grad_enabled(False):
                preds = self.model(inputs)
                loss_value = self.loss(preds, labels)
                preds_class = preds.argmax(dim=1)
            running_loss += loss_value.item()
            running_acc += (preds_class == labels.data).float().mean()
            val_targets.extend(labels.data.cpu().numpy())
            val_predictions.extend(torch.nn.functional.softmax(preds, dim=1).data.cpu().numpy())
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = running_acc / len(dataloader)
        self.val_loss_hist.append(epoch_loss)
        self.val_acc_hist.append(epoch_acc)
        print('VAL loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc), flush=True)
        self.val_predictions = np.array(val_predictions)
        self.val_targets = np.array(val_targets)
        if epoch_acc > self.best_val_acc:
            self.best_val_acc = epoch_acc
            self.save_model()

    def count_metrics_for_class(self, class_id):
        class1 = self.val_predictions[:, class_id]
        labels = self.val_targets
        tpr = [1]
        fpr = [1]
        plt.figure(1, figsize=(12, 6))
        for i in range(1, 10):
            p = float(i) / 10
            plt.subplot(3, 3, i)
            tp = (labels[class1 >= p] == class_id).sum()
            fp = (labels[class1 >= p] != class_id).sum()
            tn = (labels[class1 < p] != class_id).sum()
            fn = (labels[class1 < p] == class_id).sum()
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            print('Порог: {:.1f}, TP: {}, FP: {}, FN: {}, TN: {}, acc:{:.4f}, precision: {:.4f}, recall: {:.4f}, '
                  'F: {:.4f} '.format(p, tp, fp, fn, tn, (tp + tn) / (tp + tn + fp + fn), precision, recall,
                                      2 * precision * recall / (precision + recall)), flush=True)
            names = [['TP: %i ' % tp, 'FP: %i' % fp], ['FN: %i' % fn, 'TN: %i' % tn]]
            sns.heatmap([[tp, fp], [fn, tn]], vmin=0, vmax=len(self.val_targets), annot=names, fmt = '',
                        xticklabels=['$y=1$', '$y=0$'], yticklabels=['$\hat{y}=1$', '$\hat{y}=0$'])
            tpr.append(recall)
            fpr.append(fp / (fp + tn))
        tpr.append(0)
        fpr.append(0)
        plt.figure(2, figsize=(4.5, 4))
        plt.plot(fpr, tpr,  label='ROC')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title('ROC curve')
        plt.legend()
        plt.show()
        plt.pause(0.001)

    def count_metrics_for_model(self):
        f_mera = 0
        fig = 0
        pos = 0
        for class_id in range(self.num_class):
            preds_class = self.val_predictions.argmax(axis=1)
            labels = self.val_targets
            if class_id % 30 == 0:
                fig += 1
                plt.figure(fig, figsize=(18, 15))
                pos = 0
            pos += 1
            plt.subplot(5, 6, pos)
            tp = (labels[preds_class == class_id] == class_id).sum()
            fp = (labels[preds_class == class_id] != class_id).sum()
            tn = (labels[preds_class != class_id] != class_id).sum()
            fn = (labels[preds_class != class_id] == class_id).sum()
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f_mera += 2 * precision * recall / (precision + recall)
            print('Класс: {}, TP: {}, FP: {}, FN: {}, TN: {}, acc:{:.4f}, precision: {:.4f}, recall: {:.4f}, '
                  'F: {:.4f} '.format(class_id, tp, fp, fn, tn, (tp + tn) / (tp + tn + fp + fn), precision, recall,
                                      2 * precision * recall / (precision + recall)), flush=True)
            names = [['TP: %i ' % tp, 'FP: %i' % fp], ['FN: %i' % fn, 'TN: %i' % tn]]
            sns.heatmap([[tp, fp], [fn, tn]], vmin=0, vmax=len(self.val_targets), annot=names, fmt='',
                        xticklabels=['$y=1$', '$y=0$'], yticklabels=['$\hat{y}=1$', '$\hat{y}=0$'])
        plt.show()
        plt.pause(0.001)
        print('Average F: {:.4f} '.format(f_mera/self.num_class), flush=True)

    def plot_metrics(self, num_val):
        plt.figure(3, figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(self.epochs + 1), self.train_acc_hist, label='train')
        plt.plot(np.arange(0, self.epochs + 1, num_val), self.val_acc_hist, label='val')
        plt.title('Accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(np.arange(self.epochs + 1), self.train_loss_hist, label='train')
        plt.plot(np.arange(0, self.epochs + 1, num_val), self.val_loss_hist, label='val')
        plt.title('Loss')
        plt.legend()
        plt.show()
        plt.pause(0.001)

    def predict(self, data, batch_size):
        self.model.eval()
        test_predictions = []
        test_img_idx = []
        dataloader = torch.utils.data.DataLoader(data.test_dataset, batch_size=batch_size, shuffle=False,
                                                 num_workers=batch_size)
        for inputs, idx in tqdm(dataloader):
            inputs = inputs.to(self.device)
            with torch.set_grad_enabled(False):
                preds = self.model(inputs)
            test_predictions.extend(
                torch.nn.functional.softmax(preds, dim=1)[:, 1].data.cpu().numpy())
            test_img_idx.extend(idx)
        return test_img_idx, test_predictions

    def save_model(self):
        file_name = 'models/{}-{}-{:.4f}.pickle'.format(self.name, self.epochs, self.best_val_acc)
        print('saving model with name: "{}"'.format(file_name))
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)
