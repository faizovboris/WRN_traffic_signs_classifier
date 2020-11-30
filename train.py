import os
import json
import keras
import argparse
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")


from model import prepare_model, get_callbacks
from data import get_classes, get_list_of_files, create_test_index, create_synt_index, data_generator
from utils import read_index, read_index_icon

config = {
    'img': {
        'img_size': 64,
        'padding': 0
    },
    'wide_resnet': {
        'num_classes': 205,
        'metablock_depth': 4,
        'width': 4,
        'dropout': 0.6
    },
    'schedule': {
        'lr': 0.001,
        'gamma': 0.2,
        'lr_step': 2
    },
    'train_params': {
        'val_ratio': 0.01,
        'batch_size': 306,
        'different_classes_per_batch': 64,
        'steps_per_epoch': 2200,
        'num_epochs': 10,
        'snap_epoch': 1
    },
    'model_saves': {
        'weights_suffix': ''
    }
}


def train_model(train_filelist, config, classes, rare_classes):
    batch_size = config['train_params']['batch_size']
    different_classes_per_batch = config['train_params']['different_classes_per_batch']
    nb_epoch = config['train_params']['num_epochs']
    steps_per_epoch = config['train_params']['steps_per_epoch']
    img_size = config['img']['img_size']

    model = prepare_model(config)
    callbacks = get_callbacks(config)
    model.fit_generator(data_generator(train_filelist, classes, rare_classes, img_size, batch_size, different_classes_per_batch),
                        steps_per_epoch=steps_per_epoch, epochs=nb_epoch, callbacks=callbacks)
    return model


def train_head(model, index_x, index_y, classes_rare, n_neighbors):
    index_x = model.predict(index_x)[0]
    index_x_normalized = np.array([vec / np.linalg.norm(vec) for vec in index_x])
    classes_rare_ids = [classes[cls] for cls in classes_rare]
    rf_y = [1 if y in classes_rare_ids else 0 for y in index_y]

    rf_outclass = RandomForestClassifier(n_estimators=10000, max_depth=25, n_jobs=50)
    rf_outclass.fit(index_x_normalized, rf_y)

    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(index_x, index_y)
    return rf_outclass, knn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--exp_name', type=str, required=True, help='Name of the experiment')
    parser.add_argument('--nn_model', type=str, default="model.h5", help='File for wideresnet model')
    parser.add_argument('--gpu_ids', type=str, default='5', help='Ids of gpus for training')
    parser.add_argument('--real_path', type=str, default='data/cropped-train',
                        help='Type to real data, use \'\' to train only on synthetic data')
    parser.add_argument('--synt_path', type=str, default='data/cropped_stylegan',
                        help='Path to sytnhetic data, use \'\' to train only on real data')
    parser.add_argument('--test_path', type=str, default='data/cropped-test',
                        help='Path to real test data, need only to create index from test')
    parser.add_argument('--icons_path', type=str, default='data/icons',
                        help='Path to icons, need only to create index from icons')
    parser.add_argument('--train_type', choices=['all_classes', 'often_classes'], help='Experiment type')
    parser.add_argument('--use_existing_index', action='store_true', help='Do not create new index')
    parser.add_argument('--train_only_heads', action='store_true', help='Train only heads')
    parser.add_argument('--replication_factor', type=int, default=15, help='Replication factor for index')
    parser.add_argument('--heads', type=str, default='[["icon", 1, 1], ["test", 1, 1], ["synt", 1, 1], ["synt", 5, 3], ["synt", 10, 5]]',
                        help='Necessary heads with random forest and kNN in format \'[(index_type, index_pictures_per_class, knn_neighbors)]\'')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_ids

    classes, classes_rare, classes_often = get_classes(args.real_path)

    data_folders = []
    if args.real_path != '':
        data_folders.append(args.real_path)
    if args.synt_path != '':
        data_folders.append(args.synt_path)
    train_filelist = get_list_of_files(args.exp_name, data_folders, classes_often + classes_rare if args.train_type == 'all_classes' else classes_often)

    if args.train_only_heads:
        model = prepare_model(config)
        model.load_weights(args.exp_name + "_" + args.nn_model)
    else:
        model = train_model(train_filelist, config, classes, classes_rare)
        model.save(args.exp_name + "_" + args.nn_model)

    img_shape = (config['img']['img_size'], config['img']['img_size'], 3)

    for head in json.loads(args.heads):
        print("Training head: ", head)
        if head[0] == 'icon':
            index_x, index_y = read_index_icon(args.icons_path, img_shape, classes, args.replication_factor)
        if head[0] == 'test':
            index_file, noindex_file = create_test_index(args.exp_name, args.test_path, args.use_existing_index)
            index_x, index_y = read_index(index_file, img_shape, classes, args.replication_factor)
        if head[0] == 'synt':
            index_file = create_synt_index(args.exp_name, args.synt_path, args.use_existing_index, head[1])
            index_x, index_y = read_index(index_file, img_shape, classes, args.replication_factor)

        rf_outclass, knn = train_head(model, index_x, index_y, classes_rare, head[2])
        pickle.dump(rf_outclass, open(args.exp_name + "_rand_forest_" + '_'.join(map(str, head)) + ".bin", "wb"))
        pickle.dump(knn, open(args.exp_name + "_knn_" + '_'.join(map(str, head)) + ".bin", "wb"))



# python3 train.py --train_type='all_classes' --exp_name='testexp'


