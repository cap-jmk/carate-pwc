"""
This is the heart of the application and trains / tests a algorithm on a given dataset. 
The idea is to parametrize as much as possible. 

"""
import torch
import torch.nn.functional as F

from carate.models.cgc import Net
from carate.models.default_model import DefaultModel
from carate.utils.file_utils import check_make_dir
from carate.load_data import DataLoader, StandardDataLoaderMoleculeNet
from carate.default_interface import DefaultObject
from typing import Type


import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="example.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
)


class Evaluation(DefaultObject):

    """
    The evaluation class is about evaluating a given model written in PyTorch or PyTorchGeometric.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_save_path: str,
        model: type(DefaultModel),
        optimizer: type(torch.optim),
        device: type(torch.device),  # TODO types
        DataLoader: type(DataLoader),
        test_ratio: int,
        shrinkage: int,
        num_epoch: int = 150,
        n_cv: int = 5,
        num_classes: int = 2,
        out_dir: str = r"./out",
        gamma: int = 0.5,
        batch_size: int = 64,
        shuffle: bool = True,
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
        self.dataset_name = dataset_name
        self.dataset_save_path = dataset_save_path
        self.test_ratio = test_ratio
        self.shrinkage = shrinkage
        self.num_epoch = num_epoch
        self.model = model
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.n_cv = n_cv
        self.out_dir = out_dir
        self.gamma = gamma
        self.DataLoader = DataLoader
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

    def train(
        self,
        epoch: int,
        model: type(DefaultModel),
        device: type(torch.device),
        train_loader,  # TODO find out type
        test_loader,  # TODO find out type
        optimizer: type(torch.optim),
        num_classes=2,
        shrinkage=51,
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

    def test(self, test_loader, epoch, model, device, test=False):
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
            self.train_store = outputs
        return correct / len(loader.dataset)

    def cv(
        self,
        n_cv: int,
        num_epoch: int,
        num_classes: int,
        dataset_name: str,
        dataset_save_path: str,
        test_ratio: int,
        DataLoader: type(DataLoader),
        shuffle: bool,
        batch_size: int,
        model: type(DefaultModel),
        optimizer: type(torch.optim),
        device: type(torch.device),
        shrinkage: int,
    ):
        """
        The cv function takes in the following parameters:
            n_cv (int): The number of cross-validation folds to perform.
            num_epoch (int): The number of epochs to train for each fold.
            num_classes (int): The number of classes in the dataset.  This is used for one-hot encoding labels and calculating AUC scores.
                If you are using a dataset that has already been one-hot encoded, then this should be set to None or 1, depending on whether your data is binary or not respectively.

                For example, if you have a binary classification problem with two classes {0, 1}, then this parameter should be set to 2 because there are two possible classifications; however if your data has already been one hot encoded into {[0], [0]}, then it would only make sense for this parameter to be set as 1 since there is only one possible classification per sample point now ({[0], [0]} -> 0).

                Note that setting this value incorrectly will result in incorrect AUC scores being calculated!  It's up to you as an engineer/data scientist/machine learning practitioner/etc...to know what kind of data you're working with and how best it can be represented by PyTorch tensors!

            DataLoader: An instance of torchvision's DataLoader class which loads training and testing datasets from disk into memory so they can easily accessed during training time without having I/O overhead every time we want access our training samples!  You may need some additional arguments passed into the constructor such as batch size etc...but these details are left up to implementation specific details which will vary based on what kind of model architecture we're using etc...so I've left them out here intentionally.

        :param self: Used to Represent the instance of the class.
        :param n_cv:int: Used to Specify the number of cross-validation folds.
        :param num_epoch:int: Used to Specify the number of epochs to train for.
        :param num_classes:int: Used to Determine the number of classes in the dataset.
        :param dataset_name:str: Used to Specify the name of the dataset to be used.
        :param DataLoader:type(DataLoader): Used to Load the data.
        :param : Used to Specify the number of folds in a (stratified)kfold,.
        :return: A list of dictionaries.

        :doc-author: Trelent
        """

        (
            n_cv,
            num_epoch,
            num_classes,
            dataset_name,
            dataset_save_path,
            test_ratio,
            DataLoader,
            shuffle,
            batch_size,
            model,
            optimizer,
            device,
            shrinkage,
        ) = self._get_defaults(locals())
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
            ) = DataLoader.load_data(
                dataset_name=dataset_name,
                dataset_save_path=dataset_save_path,
                test_ratio=test_ratio,
                batch_size=batch_size,
                shuffle=shuffle,
            )

            for epoch in range(1, num_epoch):
                train_loss = self.train(
                    epoch=epoch,
                    model=model,
                    device=device,
                    optimizer=optimizer,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    num_classes=num_classes,
                    shrinkage=shrinkage,
                )
                loss_store.append(train_loss.cpu().tolist())
                train_acc = self.test(
                    train_loader, device=device, model=model, epoch=epoch
                )
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
            raise NotImplementedError  # TODO implement string methods in all classes
