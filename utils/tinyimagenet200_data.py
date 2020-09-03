import os
from shutil import copyfile
from urllib.request import urlretrieve
import zipfile


def download_dataset(path, path_old,
                     url='http://cs231n.stanford.edu/tiny-imagenet-200.zip'):
    # if not os.path.exists(path):
    #     os.mkdir(path)
    urlretrieve(url, path_old + '/tiny-imagenet-200.zip')
    # print(os.path.join(path))
    # os.makedirs(path + '/test', exist_ok=True)
    # os.makedirs(path + '/train', exist_ok=True)
    # os.makedirs(path + '/val', exist_ok=True)
    with zipfile.ZipFile(path_old + '/tiny-imagenet-200.zip', 'r') as zip_ref:
        zip_ref.extractall(os.path.join(path_old))
    zip_ref.close()


def ensure_dataset_loaded(data_path):
    val_fixed_folder = data_path + "/val_fixed"
    if os.path.exists(val_fixed_folder):
        return
    if not os.path.exists(data_path):
        print('Downloading TinyImageNet-200 Dataset.')
        download_dataset(data_path)
    os.mkdir(val_fixed_folder)

    print('Converting Validation images folder into an ImagesFolder for Torch')
    with open(data_path + "/val/val_annotations.txt") as f:
        for line in f.readlines():
            fields = line.split()

            file_name = fields[0]
            clazz = fields[1]

            class_folder = data_path + "/val_fixed/" + clazz
            if not os.path.exists(class_folder):
                os.mkdir(class_folder)

            original_image_path = data_path + "/val/images/" + file_name
            copied_image_path = class_folder + "/" + file_name

            copyfile(original_image_path, copied_image_path)