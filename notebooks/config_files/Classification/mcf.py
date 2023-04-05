dataset_name = "MCF-7"
num_classes = 2
num_features = 46
model = "cgc_classification"
evaluation = "classification"
optimizer = "adams"  # defaults to adams optimizer
net_dimension = 364
learning_rate = 0.0005
dataset_save_path = "./data"
test_ratio = 20
batch_size = 64
shuffle = True
num_epoch = 300
num_cv = 5
result_save_dir = "./MCF-7"
data_loader = "StandardTUD"
model_save_freq = 30

if __name__ == "__main__":

    from carate.run import RunInitializer

    config_filepath = "./mcf.py"
    runner = RunInitializer.from_file(config_filepath=config_filepath)
    runner.run()
