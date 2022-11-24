import torch
from model import Net 
class Run:
    """
    Write a run module
    """


    def __init__(
        self, 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
        model = Net, 
        optimizer = None, 
        net_dimension = 364,
        learning_rate = 0.0005, 
    ):  
        self.net_dimension = 364
        self.learning_rate = 0.0005,
        self.device = device
        self.model = model(dim=net_dimension).to(device)
        if optimizer is None: 
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

if __name__ == "__main__": 
    run = Run()
    print(run.model.dim)
    run.net_dimension = 50
    print(run.model.dim)
    run.__init__(net_dimension=50)
    print(run.model.dim)