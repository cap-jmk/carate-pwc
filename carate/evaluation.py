"""
This is the hear of the application and trains / tests a algorithm on a given dataset. 
The idea is to parametrize as much as possible. 

"""
from utils.file_utils import check_make_dir

class Evaluation(): 

    """
    The evaluation class is about evaluating a given model written in PyTorch or PyTorchGeometric. 


    ------Attributes------
    The evaluation has the attributes 
        epoch 
        model 
        optimizer
        num_classes
    
    """

    def __init__(self, epoch, model, optimizer, num_classes, out_dir="out"):

        self.epoch = epoch 
        self.model = model
        self.optimizer = optimizer
        self.num_classes = num_classes 
        self.out_dir = out_dir



    def train(epoch, model, device, train_loader, optimizer, num_classes = 2, shrikage=51):
        """
        The train function is used to train the model.
        It takes in an epoch number, a model, a device (cpu or gpu), and a dataset loader as input.
        The output is the accuracy of the current training iteration.

        :param epoch: Used to Determine the number of times we want to train the model.
        :param model: Used to Pass the model to the train function.
        :param device: Used to Tell the model which device to use.
        :param train_loader: Used to Pass the training data.
        :param optimizer: Used to Specify the algorithm to use for training.
        :param num_classes=2: Used to Specify the number of classes in the dataset.
        :param shrikage=51: Used to Reduce the learning rate to half after 50 epochs.
        :return: The accuracy of the model on the training set.

        :doc-author: Trelent
        """


        model.train()

        if epoch == shrinkage:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.5 * param_group['lr'] # setting the learning rate 

        correct = 0
        for data in train_loader:
            data.x = data.x.type(torch.FloatTensor)
            data.y = F.one_hot(data.y, num_classes = num_classes).type(torch.FloatTensor)
            data = data.to(device)
            optimizer.zero_grad()
            output_probs = model(data.x, data.edge_index, data.batch)
            output = (output_probs > 0.5).float()
            loss = torch.nn.BCELoss()
            loss = loss(output_probs, data.y)
            loss.backward()
            optimizer.step()
            correct += (output == data.y).float().sum()/num_classes
        accuracy = correct / len(train_loader.dataset)  
        return accuracy


    def test(loader, epoch, model, device, test=False):
        """
        The test function takes in a dataloader and model, and returns the accuracy of the model on that dataset.
        The test function also takes in an optional argument `test` which is set to False by default. 
        If this argument is set to True, then it will return all of the outputs for each example.
        
        :param loader: Used to Load the data.
        :param epoch: Used to Keep track of the current epoch.
        :param model: Used to Pass the model we want to test.
        :param device: Used to Tell torch which device to use.
        :param test=False: Used to Indicate that we are in test mode.
        :return: The accuracy of the model on the test set.
        
        :doc-author: Trelent
        """
        

        model.eval()

        correct = 0
        if test: 
            outs = []
        for data in loader:
            data.x = data.x.type(torch.FloatTensor)
            data = data.to(device)
            output_probs = model(data.x, data.edge_index, data.batch)
            output = (output_probs > 0.5).float()
            correct += (torch.argmax(output, dim=1) == data.y).float().sum()
            if test: 
                outs.append(output.cpu().detach().numpy())
        if test: 
            outputs =np.concatenate(outs, axis=0 ).astype(float)
            self.train_store=output#TODO Do not save but only store temporarily in class
        return correct / len(loader.dataset)

    def cv(data_set, n=5, num_epoch=150, num_classes = 2):
        """
        The cv function takes in a dataset name, and returns the results of cross validation.
        The function takes in a dataset name, and then splits the data into 5 folds. 
        For each fold, it trains on 4 folds and tests on 1 fold. The function also saves all 
        the metrics (losses, accuracies) for each epoch to a csv file for later analysis.
        
        :param data_set: Used to Determine which dataset to load.
        :param n=5: Used to Indicate the number of iterations.
        :param num_epoch=150: Used to Specify the number of epochs.
        :param num_classes=2: Used to Specify the number of classes in the dataset.
        :return: A list of dictionaries.
        
        :doc-author: Trelent
        """
        
        result = []
        acc_store = []
        auc_store = []
        loss_store = [] 
        tmp = {}
        for i in range(n):
            test_loader, train_loader, dataset, train_dataset, test_dataset  = load_data(dataset=data_set)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = Net(dim=364, dataset=dataset).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

            for epoch in range(1, num_epoch):
                train_loss = train(epoch=epoch, model=model, device=device, optimizer=optimizer, train_loader=train_loader, num_classes = num_classes)
                loss_store.append(train_loss.cpu().tolist())
                train_acc = test(train_loader, device=device, model=model, epoch=epoch)
                test_acc = test(test_loader, device=device, model=model, epoch=epoch, test=True)
                acc_store.append([train_acc.cpu().tolist(), test_acc.cpu().tolist()])
                print('Epoch: {:03d}, Train Loss: {:.7f}, '
                    'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                                train_acc, test_acc))
                y = np.zeros((len(test_dataset)))
                x = self.train_store
                for i in range(len(test_dataset)):
                    y[i] = test_dataset[i].y
                y = torch.as_tensor(y)
                y = F.one_hot(y.long(), num_classes = num_classes).long()
                store_auc = []
                for i in range(len(x[0,:])): 
                    auc = metrics.roc_auc_score(y[:,i], x[:,i])
                    print("AUC of "+str(i) +"is:", auc)
                    store_auc.append(auc)
                auc_store.append(store_auc)
                
                if auc >=0.9:
                    break
                tmp["Loss"] = list(loss_store)
                tmp["Acc"] = list(acc_store)
                tmp["AUC"] = auc_store
            with open(save_dir+data_set+"_"+str(i)+".csv", 'w') as f:
                json.dump(tmp, f)
                print("Saved iteration one to "+save_dir+data_set+"_"+str(i)+".csv")#TODO Parametrize 
            result.append(tmp)          
        return result 
