"""
Tests the file converter that converts a config file to a json for using to initialize a Run object.

"""

from carate.utils.convert_to_json import convert_py_to_json, read_file


def test_converter():

    ref_py = [
        "dataset_name = ZINC_test",
        "num_classes = 1",
        "num_features = 18",
        "model = CGCR",
        "device = torch.device(cuda if torch.cuda.is_available() else cpu)",
        "optimizer = None  # defaults to adams optimizer",
        "net_dimension = 364",
        "learning_rate = 0.0005",
        "dataset_save_path = data/",
        "test_ratio = 20",
        "batch_size = 64",
        "shuffle = True",
        "num_epoch = 2",
        "num_cv = 2",
        "result_save_dir = results/",
    ]
    ref_json = {
        "dataset_name": "ZINC_test",
        "num_classes": "1",
        "num_features": "18",
        "model": "CGCR",
        "device": "torch.device(cuda",
        "optimizer": None,
        "net_dimension": "364",
        "learning_rate": "0.0005",
        "dataset_save_path": "data/",
        "test_ratio": "20",
        "batch_size": "64",
        "shuffle": "True",
        "num_epoch": "2",
        "num_cv": "2",
        "result_save_dir": "results/",
    }

    input_file = "tests/config/config.py"
    output_file = "tests/config//config.json"

    raw_input_lines = read_file(file_name=input_file)  #
    assert raw_input_lines == ref_py

    result = convert_py_to_json(file_name=input_file, out_name=output_file)
    assert result == ref_json
