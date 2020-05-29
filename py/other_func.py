import os
import shutil
from tqdm import tqdm


def split_train_val(root_dir, n):
    train_dir = os.path.join('train')
    val_dir = os.path.join('val')
    class_names = os.listdir(root_dir)
    for dir_name in [train_dir, val_dir]:
        for class_name in class_names:
            os.makedirs(os.path.join(dir_name, class_name), exist_ok=True)

    for class_name in class_names:
        source_dir = os.path.join(root_dir, class_name)
        for i, file_name in enumerate(tqdm(os.listdir(source_dir))):
            if i % n != 0:
                dest_dir = os.path.join(train_dir, class_name)
            else:
                dest_dir = os.path.join(val_dir, class_name)
            shutil.copy(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))


def count_parameters(model):
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)
