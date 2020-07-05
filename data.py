import os
import glob
import numpy as np
from random import randint
from skimage.io import imread
from skimage.transform import resize
from keras.utils import to_categorical


def create_synt_index(exp_name, data_folder, use_existing_index, pic_cnt=1):
    index_file = exp_name + "_" + data_folder.split('/')[-1] + ".index"
    if use_existing_index:
        return index_file
    with open(index_file, 'w') as index:
        for class_folder in sorted(glob.glob(os.path.join(data_folder, '*'))):
            class_name = os.path.basename(class_folder)
            fnames = sorted(glob.glob(os.path.join(data_folder, class_name, '*')))
            if (pic_cnt >= len(fnames)):
                index_ids = [i for i in range(len(fnames))]
            else:
                #index_ids = np.random.choice(len(fnames), pic_cnt).tolist()
                index_ids = []
                for i in range(pic_cnt):
                    for tries_to_find_img in range(5):
                        idx = np.random.randint(0, len(fnames))
                        img = imread(fnames[idx])
                        if img.shape[0] > 30 and img.shape[1] > 30:
                            break
                    index_ids.append(idx)
            for idx in index_ids:
                print(class_name, ",", fnames[idx], "\n", end='', sep='', file=index)
    return index_file


def create_test_index(exp_name, data_folder, use_existing_index):
    index_file = exp_name + "_rtsd_test.index"
    noindex_file = exp_name + "_not_indexed_rtsd_test.files"
    if use_existing_index:
        return index_file, noindex_file
    with open(index_file, 'w') as index, open(noindex_file, 'w') as noindex:
        for class_folder in sorted(glob.glob(os.path.join(data_folder, '*'))):
            class_name = os.path.basename(class_folder)
            fnames = sorted(glob.glob(os.path.join(data_folder, class_name, '*')))
            best_id = 0
            best_size = 0
            for cur_id, path in enumerate(fnames):
                img = imread(path)
                if (img.size > best_size):
                    best_size = img.size
                    best_id = cur_id
            print(class_name, ",", fnames[cur_id], "\n", end='', sep='', file=index)
            for cur_id, path in enumerate(fnames):
                if (cur_id != best_id):
                    print(class_name, ",", fnames[cur_id], "\n", end='', sep='', file=noindex)
    return index_file, noindex_file


def get_list_of_files(exp_name, data_folders, interesting_classes): 
    file = exp_name + "_train_" + '_'.join([folder.split('/')[-1] for folder in data_folders]) + ".files"
    with open(file, 'w') as fp:
        for folder in data_folders:
            for clas_dir in sorted(glob.glob(os.path.join(folder, '*'))):
                class_name = os.path.basename(clas_dir)
                if (class_name in interesting_classes):
                    fnames = sorted(glob.glob(os.path.join(folder, class_name, '*')))
                    for path in fnames:
                        print(class_name, ",", path, "\n", end='', sep='', file=fp)
    return file


def get_classes(train_dir):
    classes = {}
    classes_often = []
    classes_rare = []
    for class_id, clas_dir in enumerate(sorted(glob.glob(os.path.join(train_dir, '*')))):
        class_name = os.path.basename(clas_dir)
        classes[class_name] = class_id
        fnames = glob.glob(os.path.join(train_dir, class_name, '*'))
        if (len(fnames) > 0):
            classes_often.append(class_name)
        else:
            classes_rare.append(class_name)
    return classes, classes_rare, classes_often


def data_generator(path, classes, rare_classes, img_size, batch_size, different_classes_per_batch):
    img_filenames = {}
    with open(path) as fp:
        for line in fp:
            cls, path_img = line.split(',')
            path_img = path_img.split('\n')[0]
            if (classes[cls] > -1):
                if (cls not in img_filenames.keys()):
                    img_filenames[cls] = []
                img_filenames[cls].append(path_img)


    X = np.zeros((batch_size, img_size, img_size, 3))
    y = np.zeros((batch_size, ), 'int64')
    zs = np.zeros((batch_size, ), 'int64')
    while (True):
        batch_pos = 0
        for i in range(different_classes_per_batch):
            cls = list(img_filenames.keys())[randint(0, len(list(img_filenames.keys())) - 1)]
            for pos in range(batch_pos, min(batch_pos + batch_size // different_classes_per_batch, batch_size)):
                i = randint(0, len(img_filenames[cls]) - 1)
                filename = img_filenames[cls][i]
                img = imread(filename)[..., :3]
                X[pos] = resize(img, (img_size, img_size))
                y[pos] = classes[cls]
                zs[pos] = 1 if cls in rare_classes else 0
            batch_pos = min(batch_pos + batch_size // different_classes_per_batch, batch_size)
        y_cat = to_categorical(y, 205)
        #y_cat_n = np.hstack((y_cat, zs[:, None]))
        #yield X.copy(), [y_cat_n, y_cat].copy()
        yield X.copy(), [y_cat, y_cat].copy()



