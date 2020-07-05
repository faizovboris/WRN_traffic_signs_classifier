import os
import argparse
import json
import pickle
import numpy as np
from skimage.io import imread
from skimage.transform import resize


from model import prepare_model
from data import get_classes, get_list_of_files, create_test_index


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
        'num_epochs': 5,
        'snap_epoch': 1
    },
    'model_saves': {
        'weights_suffix': ''
    }
}


class PredictModel():
    def __init__(self, classes, model_path, rf_path, knn_path):
        self.model = prepare_model(config)
        self.model.load_weights(model_path)
        self.rf_outclass = pickle.load(open(rf_path, 'rb'))
        self.knn = pickle.load(open(knn_path, 'rb'))

        self.classes_name_to_id = classes
        self.classes_id_to_name = {}
        for key, val in self.classes_name_to_id.items():
            self.classes_id_to_name[val] = key

    def get_class_name_by_id(self, class_id):
        return self.classes_id_to_name[class_id]
    
    def get_class_id_by_name(self, class_name):
        return self.classes_name_to_id[class_name]
            
    def predict(self, imgs):
        model_pred = self.model.predict(imgs)
        normalized_features = np.array([vec / np.linalg.norm(vec) for vec in model_pred[0]])
        indexed_clas_pred = self.knn.predict(model_pred[0])
        is_outdist_pred = self.rf_outclass.predict(normalized_features)
        model_clas_pred = model_pred[1].argmax(axis=-1)
        
        ans = []
        for pos in range(len(imgs)):
            pred_clas = model_clas_pred[pos]
            if (is_outdist_pred[pos] == 1):
                pred_clas = indexed_clas_pred[pos]
            ans.append(pred_clas)
        return ans


def load_data_from_dir(img_filenames, img_classes, img_shape, classes):
    n_imgs = len(img_filenames)
    X = np.zeros((n_imgs, ) + img_shape)
    y = np.zeros((n_imgs), 'int64')
    y_classes_names = []
    for i, filename in enumerate(img_filenames):
        img = imread(filename)[..., :3]
        X[i] = resize(img, img_shape[:2])
        cls = img_classes[i]
        y[i] = classes[cls]
        y_classes_names.append(cls)
    return X, y, y_classes_names


def test_knn(predict_model, test_dir, img_shape, classes, classes_rare, batch_size=2048):
    from keras.utils import to_categorical
    img_filenames = []
    img_classes = []
    with open(test_dir) as fp:
        for line in fp:
            cls, path_img = line.split(',')
            path_img = path_img.split('\n')[0]
            img_filenames.append(path_img)
            img_classes.append(cls)

    all_cnt = 0
    ok_cnt = 0
    all_rear_cnt = 0
    ok_rear_cnt = 0
    all_often_cnt = 0
    ok_often_cnt = 0
    for start_pos in range(0, len(img_filenames), batch_size):
        end_pos = min(len(img_classes), start_pos + batch_size)
        X, y_true, y_true_classes_names = load_data_from_dir(img_filenames[start_pos:end_pos], img_classes[start_pos:end_pos], img_shape, classes)
        y_pred = predict_model.predict(X)
        for pos in range(len(y_true)):
            all_cnt += 1
            if (y_pred[pos] == y_true[pos]):
                ok_cnt += 1
            if (y_true_classes_names[pos] in classes_rare):
                all_rear_cnt += 1
                if (y_pred[pos] == y_true[pos]):
                    ok_rear_cnt += 1
            else:
                all_often_cnt += 1
                if (y_pred[pos] == y_true[pos]):
                    ok_often_cnt += 1
    return ok_cnt / all_cnt, ok_rear_cnt / all_rear_cnt, ok_often_cnt / all_often_cnt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--exp_name', type=str, required=True, help='Name of the experiment')
    parser.add_argument('--nn_model', type=str, default="model.h5", help='File for wideresnet model')
    parser.add_argument('--gpu_ids', type=str, default='5', help='Ids of gpus for testing')
    parser.add_argument('--real_path', type=str, default='data/cropped-train',
                        help='Path to real train data, used only to get possible classes')
    parser.add_argument('--test_path', type=str, default='data/cropped-test',
                        help='Path to real test data')
    parser.add_argument('--heads', type=str, default='[["icon", 1, 1], ["test", 1, 1], ["synt", 1, 1], ["synt", 5, 3], ["synt", 10, 5]]',
                        help='Necessary heads with random forest and kNN in format \'[(index_type, index_pictures_per_class, knn_neighbors)]\'')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_ids

    classes, classes_rare, classes_often = get_classes(args.real_path)
    test_filelist = get_list_of_files(args.exp_name, [args.test_path], classes_often + classes_rare)
    img_shape = (config['img']['img_size'], config['img']['img_size'], 3)

    for head in json.loads(args.heads):
        print("Evaluating head: ", head)
        cur_test_filelist = test_filelist
        if head[0] == 'test':
            _, cur_test_filelist = create_test_index(args.exp_name, args.test_path, True)
        nn_name = args.exp_name + "_" + args.nn_model
        rf_name = args.exp_name + "_rand_forest_" + '_'.join(map(str, head)) + ".bin"
        knn_name = args.exp_name + "_knn_" + '_'.join(map(str, head)) + ".bin"
        predict_model = PredictModel(classes, nn_name, rf_name, knn_name)
        q_all, q_rare, q_often = test_knn(predict_model, cur_test_filelist, img_shape, classes, classes_rare)
        
        print("All signs: ", q_all, "  rare signs: ", q_rare, "  often signs: ", q_often)




#python3 evaluate.py --exp_name='testexp'



