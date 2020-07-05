# Classification of rare traffic signs
Our paper studies the possibility of using neural networks for the classification of objects that
are few or absent at all in the training set. The task is illustrated by the example of classification of rare traffic signs. We consider neural networks trained using a contrastive loss function and its
modifications, also we use different methods for generating synthetic samples for classification
problems. As a basic method, the indexing of classes using neural network features is used. A
comparison is made of classifiers trained with three different types of synthetic samples and their mixtures with real data. We propose a method of classification of rare traffic signs using a neural network discriminator of rare and frequent signs. The experimental evaluation shows that the proposed method allows rare traffic signs to be classified without significant loss of frequent sign classification quality. Our neural network based on WideResNet architecture.

## Info
This code is for training of classifier with simple contrastive loss. Due to lack of time we not yet prepared and published code for best pipeline with two different neural networks, first of which used to classify signs on rare and often and trained with improved contrastive loss. Even though it works slightly better, currently we don't use it in our next research works due to long train and inference time.


## Requirements
- tensorflow 1.14.0
- keras 2.3.1
- scikit-learn 0.20.0

## Datasets
RTSD dataset can be downloaded from the link http://graphics.cs.msu.ru/en/research/projects/rtsd

## Training
To train the model run script `train.py` with parameters:
- `--exp_name` - Name of the experiment
- `--nn_model` - File for wideresnet model'
- `--gpu_ids`- Ids of gpus for training
- `--real_path` - Type to real data, use '' to train only on synthetic data
- `--synt_path` - Path to sytnhetic data, use '' to train only on real data
- `--test_path` - Path to real test data, need only to create index from test
- `--icons_path` - Path to icons, need only to create index from icons
- `--train_type` - Experiment type, can be `all_classes` or `often_classes`
- `--use_existing_index` - Do not create new index
- `--replication_factor` - Replication factor for index
- `--heads` - Necessary heads with random forest and kNN in format [(index_type, index_pictures_per_class, knn_neighbors)]. Used for different types of classification.

With default parameters you cat run script by command:
> python train.py --train_type='all_classes' --exp_name='testexp'

## Evaluation
To evaluate the model run script `evaluate.py` with parameters:
- `--exp_name` - Name of the experiment
- `--nn_model` - File for wideresnet model'
- `--gpu_ids`- Ids of gpus for training
- `--real_path` - Type to real data, use '' to train only on synthetic data
- `--test_path` - Path to real test data, need only to create index from test
- `--heads` - Necessary heads with random forest and kNN in format [(index_type, index_pictures_per_class, knn_neighbors)]

With default parameters you can run script by command:
> python evaluate.py --exp_name='testexp'

## Citation
If you use this code for your research, please cite our paper [Classification of rare traffic signs ](http://www.computeroptics.smr.ru/KO/PDF/KO44-2/440213.pdf)
```
@inproceedings{rare_traffic_signs_recognition,
  title={Classification of rare traffic signs},
  author={Faizov, Boris and Shakhuro, Vlad and Sanzharov, Vadim and Konushin, Anton},
  journal={Computer Optics},
  doi = {10.18287/2412-6179-CO-601},
  pages={236--243},
  volume={44},
  number={2},
  year={2020}
}
```
