## Report for model CatBoostRegressor

**test_rmse** 60704.536919493265

**test_mae** 20516.781951163943

**test_size** 0.3

**validation_size** 0.1

I've also tried to tune some hyperparameters:
- depth [2, 10]
- iterations(number of trees) [100, 600]
- learning_rate [0.001, 1]
- l2_leaf_reg [2, 30]
- min_data_in_leaf [5, 10]

The rmse for these parameters were from 90000 to 69212.

The best combinations of params are:
- learning_rate 0.04
- depth 6
- number_of_trees 179
- min_data_in_leaf 10

## Report for model NeuralNetwork

**test_rmse** 64632.55

**test_mae** 23381.666.91942868

**test_size** 0.3

**validation_size** 0.1

I've also tried to tune some hyperparameters:
- number of hidden layers [1, 5]
- batch size [5, 512]
- optimizer
- dropout rate
- number of epochs

The rmse for these parameters were from 90000 to 69212.

The best combinations of params are:
- number of hidden layers 2
- optimizer Adam
- epochs 50
- batch size 512

## Report for model DecisionTreeRegressor
This model is characterized by a great overfiting

**test_rmse** 116492.563385

**test_mae** 24513.52321780308

**test_size** 0.3

**validation_size** 0.1

I've also tried to tune some hyperparameters:
- max_depth [2, 20]
- min_samples_split [2, 10]
- min_samples_leaf [1, 40]

The best combinations of params are:
- max_depth 14
- min_samples_leaf 37
- min_samples_slit 2


## Report for model StackedModel

The model consists of neural network and gradient boosting

**test_rmse** 59212.86287519116

**test_mae** 18513.52321780308

**test_size** 0.3