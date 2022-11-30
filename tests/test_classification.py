"""
Tests about the classficiation abilities of the models 

@author: Julian M. Kleber
"""

from run import Run
from test_loading import data_set_classification


def test_classification(data_set_classification):
    data_set_name = "ENZYMES"
    num_classes = 6
    num_features = 3
    model = Net
    device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
    optimizer = None #defaults to adams optimizer 
    net_dimension = 364
    learning_rate = 0.0005
    data_set_save_path = "."
    test_ratio = 10
    batch_size = 64
    runner = Run(  data_set_name= data_set_name,
                num_features=num_features, 
                num_classes=num_classes,
                model = model,
                device= device,
                optimizer = optimizer,
                net_dimension = net_dimension,
                learning_rate = learning_rate,
                data_set_save_path = data_set_save_path,
                test_ratio = test_ratio,
                batch_size = batch_size
            )
    runner.run()