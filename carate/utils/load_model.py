import torch



def load_model(model_path:str, model_params_path:str,model_net:type(torch.nn.Module)): 
    """
    The load_model function takes in a model_path, model_params_path and the type of network to be loaded.
    It then loads the parameters from the params file into a dictionary and uses that to create an instance of 
    the specified network. It then loads in the state dict from PATH and sets it as eval mode.
    
    :param model_path:str: Used to specify the path to the model file.
    :param model_params_path:str: Used to load the model parameters from a file.
    :param model_net:type(torch.nn.Module): Used to specify the type of model that is being loaded.
    :return: A model that is loaded with the parameters in the path.
    
    :doc-author: Julian M. Kleber
    """
    
    parameters = load_model_parameters(model_params_path)
    model = model_net(**parameters)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    return model