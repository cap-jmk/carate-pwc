dataset_name = "MCF-7"
num_classes = 2
num_features = 46
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
num_cv = 5
result_save_dir = "/media/hdd/reproduce_carate_paper/MCF-7"
log_save_dir = "/media/hdd/reproduce_carate_paper/MCF-7"
data_set = "StandardTUD"
model_save_freq = 50
override = True
device = "cpu"
heads = 3