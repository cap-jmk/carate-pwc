"""
Evaulation object for classification
"""
import torch
import numpy as np
import numpy.typing as npt
from typing import Type, Optional, Any, Tuple, Dict

from carate.evaluation.base import Evaluation
from carate.load_data import DatasetObject, StandardPytorchGeometricDataset
from carate.utils.model_files import save_model_parameters, get_latest_checkpoint
from carate.models.base_model import Model

# TODO Logging done right
import logging

logging.basicConfig(
    filename="train.log",
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
        model_net: Model,
        optimizer: torch.optim.Optimizer,
        data_set: DatasetObject,
        test_ratio: int,
        num_epoch: int = 150,
        num_cv: int = 5,
        num_classes: int = 2,
        out_dir: str = r"./out",
        batch_size: int = 64,
        shuffle: bool = True,
        model_save_freq: int = 100,
        override: bool = True,
    ):
        """

        :param self: Used to Refer to the object instance itself, and is used to access variables that belongs to the class.
        :param model: Used to Specify the model that will be trained.
        :param optimizer: Used to Define the optimizer that will be used to train the model.
        :param data_set:Type[DatasetObject]: Used to Specify the type of data loader that is used. Is implemented according to
                                             the interface given in load_data.py by the class Dataset.load_data().

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
        self.num_epoch = num_epoch
        self.model_net = model_net
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.num_cv = num_cv
        self.out_dir = out_dir
        self.data_set = data_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.result_save_dir = result_save_dir
        self.model_save_freq = model_save_freq
        self.override = override

    def cv(
        self,
        num_cv: int,
        num_epoch: int,
        num_classes: int,
        dataset_name: str,
        dataset_save_path: str,
        test_ratio: int,
        data_set: DatasetObject,
        shuffle: bool,
        batch_size: int,
        model_net: Model,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        result_save_dir: str,
        model_save_freq: int,
        override: bool = True,
    ) -> Dict[str, Any]:

        # initialize
        (
            num_cv,
            num_epoch,
            num_classes,
            dataset_name,
            dataset_save_path,
            test_ratio,
            data_set,
            shuffle,
            batch_size,
            model_net,
            optimizer,
            device,
            result_save_dir,
            model_save_freq,
            override,
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
            loaded_dataset: torch.utils.data.Dataset
            (
                train_loader,
                test_loader,
                loaded_data_set,
                train_dataset,
                test_dataset,
            ) = data_set.load_data(
                dataset_name=dataset_name,
                dataset_save_path=dataset_save_path,
                test_ratio=test_ratio,
                batch_size=batch_size,
                shuffle=shuffle,
            )

            norm_factor = self.__normalization_factor(
                data_set=loaded_data_set, num_classes=num_classes
            )
            for epoch in range(1, num_epoch + 1):
                train_mae_loss = self.train(
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
                )
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

                if epoch % model_save_freq == 0:
                    self.save_whole_checkpoint(
                        result_save_dir=result_save_dir,
                        dataset_name=dataset_name,
                        num_cv=i,
                        num_epoch=epoch,
                        model_net=model_net,
                        data=tmp,
                        optimizer=optimizer,
                        loss=train_mae_loss,
                        override=override,
                    )

            result[str(i)] = tmp
        return result

    def train(
        self,
        epoch: int,
        model_net: Model,
        device: torch.device,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        num_classes: int,
        **kwargs: Any,
    ) -> float:
        norm_factor = float(kwargs["norm_factor"])
        model_net.train()
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
        test_loader: Type[torch.utils.data.DataLoader],
        epoch: int,
        model_net: Model,
        device: torch.device,
        **kwargs: Any,
    ) -> Tuple[float, float]:
        norm_factor = float(kwargs["norm_factor"])
        model_net.eval()
        mae = 0
        mse = 0
        for data in test_loader:
            data.x = data.x.type(torch.FloatTensor)
            data.y = data.y / norm_factor
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

    def __normalization_factor(
        self, data_set: Any, num_classes: int
    ) -> npt.NDArray[np.float64]:

        y = np.zeros((len(data_set), 1, num_classes))
        for i in range(len(data_set)):
            y[i, :, :] = data_set[i].y
        norm_factor = np.zeros((num_classes))
        for i in range(num_classes):
            norm = np.linalg.norm(y[:, 0, i], ord=2)
            norm_factor[i] = norm
        return norm_factor

    def __repr__(self) -> str:
        return "Regression Evaluation Object"
