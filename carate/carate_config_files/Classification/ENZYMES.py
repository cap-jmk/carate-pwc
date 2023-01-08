dataset_name = "ENZYMES"
num_classes = 6
num_features = 3
model = "cgc_classification"
evaluation = "classification"
optimizer = "adams"  # defaults to adams optimizer
net_dimension = 364
learning_rate = 0.0005
dataset_save_path = "data/"
test_ratio = 10
batch_size = 64
shuffle = True
gamma = 51
num_epoch = 5000
num_cv = 5
result_save_dir = "./ENZYMES_10"
data_loader = "StandardTUD"
model_save_freq = 30

if __name__ == "__main__":

    from carate.run import Run

    config_filepath = "./ENZYMES.py"
    runner = Run.from_file(config_filepath=config_filepath)
    runner.run()
