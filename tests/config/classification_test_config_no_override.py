dataset_name = "ENZYMES"
num_classes = 6
num_features = 3
model = "cgc_classification"
evaluation = "classification"
optimizer = "adams"  # defaults to adams optimizer
net_dimension = 364
learning_rate = 0.0005
dataset_save_path = "tests/data/"
test_ratio = 10
batch_size = 64
shuffle = True
num_epoch = 2
num_cv = 2
result_save_dir = "tests/results/ENZYMES"
data_set = "StandardTUD"
model_save_freq = 1
override = False
