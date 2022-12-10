"""
Evaulation object for classification
"""
import torch
import numpy as np

from carate.evaluation.evaluation import Evaluation
from carate.load_data import DataLoader


# TODO Logging done right
import logging

logging.basicConfig(
    filename="example.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
)


class RegressionEvaluation(Evaluation):
    def __init__(
        self,
        dataset_name: str,
        dataset_save_path: str,
        result_save_dir: str,
        model_net: type(torch.nn.Module),  # TODO type should be the correct model
        optimizer: type(torch.optim),
        device: type(torch.device),
        DataLoader: type(DataLoader),
        test_ratio: int,
        shrinkage: int,
        num_epoch: int = 150,
        n_cv: int = 5,
        num_classes: int = 2,
        out_dir: str = r"./out",
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
        :return: The following:.

        :doc-author: Julian M. Kleber
        """
        self.dataset_name = dataset_name
        self.dataset_save_path = dataset_save_path
        self.test_ratio = test_ratio
        self.shrinkage = shrinkage
        self.num_epoch = num_epoch
        self.model_net = model_net
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.n_cv = n_cv
        self.out_dir = out_dir
        self.DataLoader = DataLoader
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.result_save_dir = result_save_dir

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
        model_net: type(torch.nn.Module),
        optimizer: type(torch.optim),
        device: type(torch.device),
        shrinkage: int,
        result_save_dir: str,
    ):

        # initialize
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
            result_save_dir,
        ) = self._get_defaults(locals())

        # data container
        result = {}
        test_mae = []
        test_mse = []
        train_mae = []
        train_mse = []
        tmp = {}

        for i in range(n):

            factor = __normalization_factor(data_set=dataset, num_classes=num_classes)
            for epoch in range(1, num_epoch):
                mae = train(
                    model=model,
                    epoch=epoch,
                    optimizer=optimizer,
                    train_loader=train_loader,
                    device=device,
                    factor=factor,
                    num_classes=num_classes,
                )
                train_mae_val, train_mse_val = test(
                    model,
                    loader=train_loader,
                    device=device,
                    factor=factor,
                    num_classes=num_classes,
                )
                test_mae_val, test_mse_val = test(
                    model,
                    loader=test_loader,
                    device=device,
                    factor=factor,
                    num_classes=num_classes,
                )
                train_mae.append(train_mae_val)
                train_mse.append(train_mse_val)
                test_mse.append(test_mae_val)
                test_mse.append(test_mse_val)

                LOGGER.info(
                    "Epoch: {:03d}, Train MAE, MSE at epoch: ({:.7f}, {:.7f}), Test MAE, MSE at epoch: ({:.7f}, {:.7f})".format(
                        epoch, train_mae_val, train_mse_val, test_mae_val, test_mse_val
                    )
                )
                torch.cuda.empty_cache()

            tmp["MAE Train"] = list(train_mae)
            tmp["MSE Train"] = list(train_mse)
            tmp["MAE Test"] = list(train_mae)
            tmp["MSE Test"] = list(train_mse)
            result[str(i)] = tmp
            __save_result(dataset=data_set, result=result)
        return result

    # TODO the functions actually need default initialization
    # TODO implement their own training and test function
    def train(
        self,
        epoch: int,
        model_net,
        norm_factor: int,
        device,
        train_loader,
        optimizer,
        num_classes=2,
    ):
        model.train()

        if epoch == 51:
            for param_group in optimizer.param_groups:
                param_group["lr"] = 0.5 * param_group["lr"]

        mae = 0
        for data in train_loader:
            data.x = data.x.type(torch.FloatTensor)
            data.y = data.y / factor[0]
            data.y = torch.nan_to_num(data.y.type(torch.FloatTensor))
            data = data.to(device)
            optimizer.zero_grad()
            output_probs = model(data.x, data.edge_index, data.batch).flatten()
            loss = torch.nn.MSELoss()  # TODO either return or delete
            loss = loss(output_probs, data.y)  # TODO either return or delete
            loss_mae = torch.nn.L1Loss()
            mae += loss_mae(output_probs, data.y).item()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()  # TODO verify if necessary
        return mae / len(train_loader)

    def test(
        self,
        test_loader,
        epoch: int,
        model_net,
        device: type(troch.device),
    ):

        model.eval()
        mae = 0
        for data in loader:
            data.x = data.x.type(torch.FloatTensor)
            data.y = data.y / factor[0]
            data.y = torch.nan_to_num(data.y.type(torch.FloatTensor))
            data = data.to(device)
            output_probs = model(data.x, data.edge_index, data.batch)
            loss_mae = torch.nn.L1Loss()
            mae += loss_mae(output_probs, data.y).item()
            torch.cuda.empty_cache()  # TOOD verify if necessary
        return mae / len(loader)

    def __normaliuation_factor(self, dataset, num_classes: int):

        y = np.zeros((len(data_set), 1, num_classes))
        for i in range(len(data_set)):
            y[i, :, :] = data_set[i].y
        factor = np.zeros((num_classes))
        for i in range(num_classes):
            norm = np.linalg.norm(y[:, 0, i], ord=2)
            factor[i] = norm
        return factor

    def __save_result(dataset, result):

        import csv

        with open(
            "/content/drive/MyDrive/CARATE_RESULTS/" + dataset + "_20split.csv", "w"
        ) as f:
            w = csv.writer(f)
            for k, v in result.items():
                w.writerow([k, v])
