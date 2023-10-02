dataset_name = "MOLT-4"
num_classes = 2
num_features = 64
model = "cgc_classification"
evaluation = "classification"
optimizer = "adams"  # defaults to adams optimizer
net_dimension = 364
learning_rate = 0.0005
dataset_save_path = "/media/hdd/reproduce_carate_paper"
test_ratio = 20
batch_size = 64
shuffle = True
num_epoch = 500
num_cv = 1
result_save_dir = "/media/hdd/reproduce_carate_paper/MOLT-4"
log_save_dir = "/media/hdd/reproduce_carate_paper/MOLT-4"
data_set = "StandardTUD"
model_save_freq = 50
override = True
device = "cpu"
heads = 3