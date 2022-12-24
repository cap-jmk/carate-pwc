"""
Evaulation object for classification
"""
import torch
import numpy as np

from carate.evaluation.evaluation import Evaluation
from carate.load_data import DataLoader
from carate.utils.model_files import save_model_parameters

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
        # TODO type should be the correct model
        model_net: type(torch.nn.Module),
        optimizer: type(torch.optim),
        device: type(torch.device),
        DataLoader: type(DataLoader),
        test_ratio: int,
        shrinkage: int,
        num_epoch: int = 150,
        num_cv: int = 5,
        num_classes: int = 2,
        out_dir: str = r"./out",
        batch_size: int = 64,
        shuffle: bool = True,
        model_save_freq: int = 100,
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
        self.num_cv = num_cv
        self.out_dir = out_dir
        self.DataLoader = DataLoader
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.result_save_dir = result_save_dir
        self.model_save_freq = model_save_freq

    def cv(
        self,
        num_cv: int = None,
        num_epoch: int = None,
        num_classes: int = None,
        dataset_name: str = None,
        dataset_save_path: str = None,
        test_ratio: int = None,
        DataLoader: type(DataLoader) = None,
        shuffle: bool = None,
        batch_size: int = None,
        model_net: type(torch.nn.Module) = None,
        optimizer: type(torch.optim) = None,
        device: type(torch.device) = None,
        shrinkage: int = None,
        result_save_dir: str = None,
        model_save_freq: int = None,
    ):

        # initialize
        (
            num_cv,
            num_epoch,
            num_classes,
            dataset_name,
            dataset_save_path,
            test_ratio,
            DataLoader,
            shuffle,
            batch_size,
            model_net,
            optimizer,
            device,
            shrinkage,
            result_save_dir,
            model_save_freq,
        ) = self._get_defaults(locals())

        # data container
        result = {}
        test_mae = []
        test_mse = []
        train_mae = []
        train_mse = []
        tmp = {}

        save_model_parameters(model_net=model_net, save_dir=result_save_dir)
        for i in range(num_cv):

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

            norm_factor = self.__normalization_factor(
                dataset=dataset, num_classes=num_classes
            )
            for epoch in range(1, num_epoch + 1):
                mae = self.train(
                    model_net=model_net,
                    epoch=epoch,
                    optimizer=optimizer,
                    train_loader=train_loader,
                    device=device,
                    norm_factor=norm_factor,
                    num_classes=num_classes,
                )
                train_mae_val, train_mse_val = self.test(
                    model_net=model_net,
                    test_loader=train_loader,
                    device=device,
                    norm_factor=norm_factor,
                    epoch=epoch,
                )  # TODO might be unneccessary
                test_mae_val, test_mse_val = self.test(
                    model_net=model_net,
                    test_loader=test_loader,
                    device=device,
                    norm_factor=norm_factor,
                    epoch=epoch,
                )
                train_mae.append(train_mae_val)
                train_mse.append(train_mse_val)
                test_mse.append(test_mae_val)
                test_mse.append(test_mse_val)
                self.save_model_checkpoint(
                    model_save_freq=model_save_freq,
                    result_save_dir=result_save_dir,
                    dataset_name=dataset_name,
                    num_cv=i,
                    num_epoch=epoch,
                    model_net=model_net,
                )
                logging.info(
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
            self.save_result(
                result_save_dir=result_save_dir,
                dataset_name=dataset_name,
                num_cv=i,
                data=tmp,
            )
        return result

    # TODO the functions actually need default initialization
    # TODO implement their own training and test function
    def train(
        self,
        epoch: int,
        model_net: type(torch.nn.Module),
        norm_factor: float,
        device: type(torch.device),
        train_loader,
        optimizer: type(torch.optim),
        num_classes: int,
    ) -> float:

        model_net.train()  # TODO deleted shrinkage block due to minor influence
        mse = 0
        for data in train_loader:
            data.x = data.x.type(torch.FloatTensor)
            data.y = (data.y.numpy()) / norm_factor
            data.y = torch.from_numpy(data.y).type(torch.FloatTensor)
            data.y = torch.nan_to_num(data.y.type(torch.FloatTensor))
            data = data.to(device)
            optimizer.zero_grad()
            output_probs = model_net(data.x, data.edge_index, data.batch)
            loss = torch.nn.MSELoss()
            if len(data.y.size()) == 1:
                loss = loss(output_probs, data.y[:])
            else:
                loss = loss(output_probs, data.y[:, :])
            mse += loss.item()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        return mse / len(train_loader)

    def test(
        self,
        test_loader,
        epoch: int,
        norm_factor: float,
        model_net: type(torch.nn.Module),
        device: type(torch.device),
    ):

        model_net.eval()
        mae = 0
        mse = 0
        for data in test_loader:
            data.x = data.x.type(torch.FloatTensor)
            data.y = data.y / norm_factor[0]
            data.y = torch.nan_to_num(data.y.type(torch.FloatTensor))
            data = data.to(device)
            output_probs = model_net(data.x, data.edge_index, data.batch)
            loss_mae = torch.nn.L1Loss()
            mae += loss_mae(output_probs, data.y).item()
            loss = torch.nn.MSELoss()
            mse += loss_mae(output_probs, data.y).item()
            torch.cuda.empty_cache()
        return mae / len(test_loader), mse / len(
            test_loader
        )  # TODO verify if necessary

    def __normalization_factor(self, dataset, num_classes: int):

        y = np.zeros((len(dataset), 1, num_classes))
        for i in range(len(dataset)):
            y[i, :, :] = dataset[i].y
        norm_factor = np.zeros((num_classes))
        for i in range(num_classes):
            norm = np.linalg.norm(y[:, 0, i], ord=2)
            norm_factor[i] = norm
        return norm_factor
