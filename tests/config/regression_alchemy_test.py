dataset_name = "alchemy_full"
num_classes = 12
num_features = 6
model = "cgc_regression"
evaluation = "regression"
optimizer = "adams"  # defaults to adams optimizer
net_dimension = 364
learning_rate = 0.0005
dataset_save_path = "tests/data/"
test_ratio = 20
batch_size = 64
shuffle = True
num_epoch = 2
num_cv = 2
result_save_dir = "tests/results/ALCHEMY_test"
data_set = "StandardTUD"
model_save_freq = 1
override = False
device = "auto"
normalize = True
custom_size = 100
