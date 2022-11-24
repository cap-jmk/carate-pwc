
from model import Net 
class Run:

    learning_rate = 0.0005
    net_dimension = 364

    def __init__(
        self, 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
        model = Net(dim=net_dimension).to(device), 
        optimizer = torch.optim.Adam(model.parameters()), 
        net_dimension = self.net_dimension,
        learning_rate = self.learning_rate, 
    ):
        self.device = device
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if __name__ == "__main__": 
    run = Run()
    print(run.model.dim)
    run.net_dimension = 50
    print(run.model.dim)
    run.__init__()
    print(run.model.dim)