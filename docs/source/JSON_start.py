from carate.runner.run import RunInitializer

# Set Parameters
parameters = {
"dataset_name" : "PROTEINS",
"num_classes" : 2,
"num_features" : 3,
"model" : "cgc_classification",
"evaluation" : "classification",
"optimizer" : "adams",  # defaults to adams optimizer
"net_dimension" : 364,
"learning_rate" : 0.0005,
"dataset_save_path" : "./data",
"test_ratio" : 20,
"batch_size" : 64,
"shuffle" : True,
"num_epoch" : 10,
"num_cv" : 1,
"result_save_dir" : "./PROTEINS_20",
"data_set" : "StandardTUD",
"model_save_freq" : 5, 
"device": "cpu",
"override": True, 
}


#intialize a Run object

runner = RunInitializer.from_json(json_object=parameters)
runner.run()