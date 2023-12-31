dataset_name = "alchemy_full"
num_classes = 12
num_features = 6
model = "cgc_regression"
evaluation = "regression"
optimizer = "adams"  # defaults to adams optimizer
net_dimension = 364
learning_rate = 0.0005
dataset_save_path = "/media/hdd/reproduce_carate_paper/"
test_ratio = 20
batch_size = 64
shuffle = True
num_epoch = 150
num_cv = 5
result_save_dir = "/media/hdd/reproduce_carate_paper/ALCHEMY_20_no_norm"
log_save_dir = "/media/hdd/reproduce_carate_paper/ALCHEMY_20_no_norm"
data_set = "StandardTUD"
model_save_freq = 15
override = True
device = "cpu"
normalize = False
heads = 3