"""
This is the heart of the application and trains / tests a algorithm on a given dataset. 
The idea is to parametrize as much as possible. 

"""
from carate.utils.file_utils import check_make_dir
from carate.load_data import DataLoader, StandardDataLoader

from typing import Type

import logging 
logger = logging.getLogger(__name__)
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s %(message)s')


class Evaluation:

    """
    The evaluation class is about evaluating a given model written in PyTorch or PyTorchGeometric.
    """

    def __init__(
        self,
        model,
        optimizer,
        data_loader: type(DataLoader.load_data),
        epoch: int = 150,
        num_cv: int = 5,
        num_classes: int = 2,
        out_dir: str = r"./out",
        gamma: int = 0.5,
    ):
        """

        :param self: Used to Refer to the object instance itself, and is used to access variables that belongs to the class.
        :param model: Used to Specify the model that will be trained.
        :param optimizer: Used to Define the optimizer that will be used to train the model.
        :param data_loader:Type(DataLoader): Used to Specify the type of data loader that is used. Is implemented according to
                                             the interface given in load_data.py by the class DataLoader.load_data().

        :param epoch:int=150: Used to Set the number of epochs to train for.
        :param num_cv:int=5: Used to Specify the number of cross validations that will be used in the training process.
        :param num_classes:int=2: Used to Define the number of classes in the dataset.
        :param out_dir:str="out": Used to Specify the directory where the output of your training will be stored.
        :param gamma=0.5: Used to Set the decay rate of the loss function.
        :return: The following:.

        :doc-author: Julian M. Kleber
        """

        self.epoch = epoch
        self.model = model
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.out_dir = out_dir
        self.gamma = gamma
        self.dataLoader = DataLoader

    def train(
        epoch: int,
        model,
        device,
        train_loader,
        test_loader,
        optimizer,
        num_classes=2,
        shrikage=51,
    ):
        """
        The train function is used to train the model.
           The function takes in a number of epochs and a model, and returns the accuracy on the test set.

        :param epoch: Used to Determine when to stop training.
        :param model: Used to Pass the model to the function.
        :param device: Used to Tell the model which device to use.
        :param train_loader: Used to Load the training data.
        :param test_loader: Used to Evaluate the model on the test data.
        :param optimizer: Used to Specify the optimizer that will be used in training.
        :param num_classes=2: Used to Specify the number of classes in the data.
        :param shrikage=51: Used to Make sure that the model is trained for at least 51 epochs.
        :return: The accuracy of the model on the training set.

        :doc-author: Trelent
        """

        model.train()

        if epoch == shrinkage:
            for param_group in optimizer.param_groups:
                param_group["lr"] = (
                    gamma * param_group["lr"]
                )  # setting the learning rate behaviour over time

        correct = 0
        for data in train_loader:
            data.x = data.x.type(torch.FloatTensor)
            data.y = F.one_hot(data.y, num_classes=num_classes).type(torch.FloatTensor)
            data = data.to(device)
            optimizer.zero_grad()
            output_probs = model(data.x, data.edge_index, data.batch)
            output = (output_probs > 0.5).float()
            loss = torch.nn.BCELoss()
            loss = loss(output_probs, data.y)
            loss.backward()
            optimizer.step()
            correct += (output == data.y).float().sum() / num_classes
        accuracy = correct / len(train_loader.dataset)
        return accuracy

    def test(test_loader, epoch, model, device, test=False):
        """
        The test function is used to test the model on a dataset.
        It returns the accuracy of the model on that dataset

        :param test_loader: Used to Pass the test data loader.
        :param epoch: Used to Keep track of the current epoch.
        :param model: Used to Pass the model to the test function.
        :param device: Used to Tell torch which device to use.
        :param test=False: Used to Distinguish between training and testing.
        :return: The accuracy of the model on the test data.

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
            outputs = np.concatenate(outs, axis=0).astype(float)
            self.train_store = (
                outputs 
            )
        return correct / len(loader.dataset)

    def cv(
        n_cv: int,
        num_epoch: int,
        num_classes: int,
        DataLoader: type(DataLoader),
    ):
        """
        The cv function takes in a dataset name, and returns the results of cross validation.
        The function takes in a dataset name, and then splits the data into 5 folds.
        For each fold, it trains on 4 folds and tests on 1 fold. The function also saves all
        the metrics (losses, accuracies) for each epoch to a csv file for later analysis.

        :param data_set: Used to Determine which dataset to load.
        :param n_cv: Used to Indicate the number of iterations.
        :param num_epoch: Used to specify the number of epochs.
        :param num_classes: Used to specify the number of classes in the dataset.
        :return: A list of dictionaries.

        :doc-author: Trelent
        """
       
        n_cv, num_epoch, num_classes, DataLoader = self._get_defaults(locals())

        result = []
        acc_store = []
        auc_store = []
        loss_store = []
        tmp = {}
        for i in range(n_cv):
            (   
                train_loader,
                test_loader,
                dataset,
                train_dataset,
                test_dataset,
            ) = DataLoader.load_data()

            for epoch in range(1, num_epoch):
                train_loss = self.train(
                    epoch=epoch,
                    model=model,
                    device=device,
                    optimizer=optimizer,
                    train_loader=train_loader,
                    num_classes=num_classes,
                )
                loss_store.append(train_loss.cpu().tolist())
                train_acc = self.test(train_loader, device=device, model=model, epoch=epoch)
                test_acc = self.test(
                    test_loader, device=device, model=model, epoch=epoch, test=True
                )
                acc_store.append([train_acc.cpu().tolist(), test_acc.cpu().tolist()])
                print(
                    "Epoch: {:03d}, Train Loss: {:.7f}, "
                    "Train Acc: {:.7f}, Test Acc: {:.7f}".format(
                        epoch, train_loss, train_acc, test_acc
                    )
                )
                y = np.zeros((len(test_dataset)))
                x = self.train_store
                for i in range(len(test_dataset)):
                    y[i] = test_dataset[i].y
                y = torch.as_tensor(y)
                y = F.one_hot(y.long(), num_classes=num_classes).long()
                store_auc = []
                for i in range(len(x[0, :])):
                    auc = metrics.roc_auc_score(y[:, i], x[:, i])
                    print("AUC of " + str(i) + "is:", auc)
                    store_auc.append(auc)
                auc_store.append(store_auc)

                if auc >= 0.9:
                    break
                tmp["Loss"] = list(loss_store)
                tmp["Acc"] = list(acc_store)
                tmp["AUC"] = auc_store
            with open(save_dir + data_set + "_" + str(i) + ".csv", "w") as f:
                json.dump(tmp, f)
                logging.INFO(
                    "Saved iteration one to "
                    + save_dir
                    + data_set
                    + "_"
                    + str(i)
                    + ".csv"
                )
            result.append(tmp)
        return result

        def __str__(self): 
            raise NotImplementedError #TODO implement string methods in all classes 