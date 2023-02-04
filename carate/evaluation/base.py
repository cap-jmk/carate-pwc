"""
This is the heart of the application and trains / tests a algorithm on a given dataset.
The idea is to parametrize as much as possible.

:author: Julian M. Kleber
"""
from typing import Type, Any, Dict
import json
import numpy as np

import logging

from sklearn import metrics
import torch
import torch.nn.functional as F
from amarium.utils import check_make_dir, prepare_file_name_saving

from carate.utils.model_files import (
    save_model_training_checkpoint,
    save_model_parameters,
    load_model_training_checkpoint,
)

from carate.load_data import (
    DatasetObject,
)
from carate.default_interface import DefaultObject
from carate.models.base_model import Model


logging.basicConfig(
    filename="train.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
)


class Evaluation(DefaultObject):

    """
    The evaluation class is about evaluating a given model written in PyTorch or PyTorchGeometric.
    """

    name = "Default evaluation"

    def __init__(
        self,
        dataset_name: str,
        dataset_save_path: str,
        result_save_dir: str,
        model_net: Model,
        optimizer: torch.optim.Optimizer,
        data_set: DatasetObject,
        device: torch.device,
        test_ratio: int,
        num_epoch: int = 150,
        num_cv: int = 5,
        num_classes: int = 2,
        out_dir: str = r"./out",
        batch_size: int = 64,
        shuffle: bool = True,
        model_save_freq: int = 100,
        override: bool = True,
    ) -> None:
        """

        :param self: Used to Refer to the object instance itself, and is used to
        access variables that belongs to the class.
        :param model: Used to Specify the model that will be trained.
        :param optimizer: Used to Define the optimizer that will be used
        to train the model.
        :param data_set:Type[DatasetObject]: Used to Specify the type of data loader
        that is used. Is implemented according to the interface given in load_data.py
        by the class DatasetObject.load_data().

        :param epoch:int=150: Used to Set the number of epochs to train for.
        :param num_cv:int=5: Used to Specify the number of cross validations
        that will be used in the training process.
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
        self.device = device
        self.result_save_dir = result_save_dir
        self.model_save_freq = model_save_freq
        self.override = override
        self.train_store = None

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
        """
        The function is the core of the evaluation. The results are saved on disk during
        the run and returned as json at the end of the run.

        :param self: Used to Represent the instance of the class.
        :param num_cv:int: Used to Specify the number of cross-validation folds.
        :param num_epoch:int: Used to Specify the number of epochs to train for.
        :param num_classes:int: Used to Determine the number of classes in the dataset.
        :param dataset_name:str: Used to Specify the name of the dataset to be used.
        :param DataSetType[DatasetObject]: Used to Load the data.
        :param : Used to Specify the number of folds in a (stratified)kfold,.
        :return: A list of dictionaries.

        :doc-author: Trelent
        """

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
        result = []

        tmp = {}
        save_model_parameters(model_net, save_dir=result_save_dir)
        for i in range(num_cv):
            (
                train_loader,
                test_loader,
                loaded_dataset,
                train_dataset,
                test_dataset,
            ) = data_set.load_data(
                dataset_name=dataset_name,
                dataset_save_path=dataset_save_path,
                test_ratio=test_ratio,
                batch_size=batch_size,
                shuffle=shuffle,
            )
            # storage containers

            acc_store_train = []
            acc_store_test = []
            auc_store = []
            loss_store = []

            for epoch in range(1, num_epoch + 1):

                train_loss = self.train(
                    epoch=epoch,
                    model_net=model_net,
                    device=device,
                    optimizer=optimizer,
                    train_loader=train_loader,
                    num_classes=num_classes,
                )
                loss_store.append(train_loss.cpu().tolist())
                train_acc = self.test(
                    train_loader,
                    device=device,
                    model_net=model_net,
                    test=False,
                )  # test False for storing the results
                test_acc, self.train_store = self.test(
                    test_loader,
                    device=device,
                    model_net=model_net,
                    epoch=epoch,
                    test=True,
                )
                acc_store_train.append(train_acc.cpu().tolist())
                acc_store_test.append(test_acc.cpu().tolist())
                logging.info(
                    "Epoch: {:03d}, Train Loss: {:.7f}, Train Acc: {:.7f}, Test Acc: {:.7f}".format(
                        epoch, train_loss, train_acc, test_acc
                    )
                )
                preds = np.zeros((len(test_dataset)))
                features = self.train_store

                for j in range(len(test_dataset)):
                    preds[j] = test_dataset[j].y

                preds = torch.as_tensor(preds)
                preds = F.one_hot(preds.long(), num_classes=num_classes)
                store_auc = []

                for j in range(len(features[0, :])):
                    auc = metrics.roc_auc_score(preds[:, j], features[:, j])
                    logging.info("AUC of " + str(j) + "is:" + str(auc))
                    store_auc.append(auc)

                auc_store.append(store_auc)

                tmp["Loss"] = list(loss_store)
                tmp["Acc_train"] = list(acc_store_train)
                tmp["Acc_test"] = list(acc_store_test)
                tmp["AUC"] = list(auc_store)

                if epoch % model_save_freq == 0:

                    self.save_whole_checkpoint(
                        result_save_dir=result_save_dir,
                        dataset_name=dataset_name,
                        num_cv=i,
                        num_epoch=epoch,
                        model_net=model_net,
                        data=tmp,
                        optimizer=optimizer,
                        loss=train_loss,
                        override=override,
                    )
            result.append(tmp)
        return result

    def train(
        self,
        epoch: int,
        model_net: Model,
        device: torch.device,
        train_loader: Type[torch.utils.data.DataLoader],
        optimizer: torch.optim.Optimizer,
        num_classes: int,
    ):
        """
        The train function is used to train the model.
        The function takes in a number of epochs and a model, 
        and returns the accuracy on the test set.

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
        model_net.train()

        correct = 0
        for data in train_loader:
            data.x = data.x.type(torch.FloatTensor)
            data.y = F.one_hot(data.y.long(), num_classes=num_classes).type(
                torch.FloatTensor
            )
            data = data.to(device)
            optimizer.zero_grad()
            output_probs = model_net(data.x, data.edge_index, data.batch)
            output = (output_probs > 0.5).float()
            loss = torch.nn.BCELoss()
            loss = loss(output_probs, data.y)
            loss.backward()
            optimizer.step()
            correct += (output == data.y).float().sum() / num_classes
        accuracy = correct / len(train_loader.dataset)
        return accuracy

    def test(
        self,
        test_loader: torch.utils.data.DataLoader,
        model_net: Model,
        device: torch.device,
        **kwargs: Any,
    ) -> Any:
        """
        The test function is used to test the model on a dataset.
        It returns the accuracy of the model on that dataset calculated as
        the average of the atomic accuracy for each batch in the Dataset

        :param test_loader: Used to pass the test data loader.
        :param epoch: Used to keep track of the current epoch.
        :param model_net: Used to pass the model to the test function.
        :param device: Used to tell torch which device to use.
        :param test=False: Used to distinguish between training
        and testing.
        :return: The accuracy of the model on the test data.

        :doc-author: Julian M. Kleber
        """
        test = bool(kwargs["test"])
        model_net.eval()
        correct = 0
        if test:
            outs = []
        for data in test_loader:
            data.x = data.x.type(torch.FloatTensor)
            data = data.to(device)
            output_probs = model_net(data.x, data.edge_index, data.batch)
            output = (output_probs > 0.5).float()
            correct += (torch.argmax(output, dim=1) == data.y).float().sum()
            if test:
                outs.append(output.cpu().detach().numpy())
        if test:
            outputs = np.concatenate(outs, axis=0).astype(float)
            return (
                correct / len(test_loader.dataset),
                outputs,
            )  # TODO this is from some "quick experiment" not sure if the line and functionality is
        return correct / len(test_loader.dataset)

    def save_result(
        self,
        result_save_dir: str,
        dataset_name: str,
        num_cv: int,
        num_epoch: int,
        data: dict,
    ) -> None:
        """
        The save_result function saves the results of a cross-validation run to a .json file. The goal is to provide
        a json interface of cv results for later analysis of the training runs.


        :param self: Used to represent the instance of the class.
        :param result_save_dir:str: Used to specify the directory where the results will be saved.
        :param dataset_name:str: Used to identify the dataset.
        :param num_cv:int: Used to specify the number of cross validation runs.
        :param num_epoch int: Epoch the run was saved in
        :param data:dict: Used to store the results of each cross validation run.
        :return: None.

        :doc-author: Julian M. Kleber
        """

        prefix = result_save_dir + "/data/" + "CV_" + str(num_cv)
        file_name = prepare_file_name_saving(
            prefix=prefix,
            file_name=dataset_name,
            suffix=".json",
        )
        with open(file_name, "w") as f:
            json.dump(data, f)
            logging.info(
                "Saved"
                + str(num_epoch)
                + "of cv"
                + str(num_cv)
                + " run to "
                + result_save_dir
                + dataset_name
                + "_"
                + str(num_cv)
                + ".csv"
            )

    def save_whole_checkpoint(
        self,
        result_save_dir: str,
        dataset_name: str,
        num_cv: int,
        num_epoch: int,
        model_net: Type[torch.nn.Module],
        data: dict,
        optimizer: Type[torch.optim.Optimizer],
        loss: float,
        override: bool = True,
    ) -> None:
        """
        The save_whole_checkpoint function saves the model checkpoint and
        results for a given epoch.

        The save_whole_checkpoint function saves the model checkpoint
        and results for a given epoch. It is called by the train function
        in order to save checkpoints at regular intervals during training,
        as well as after each cross-validation fold has been trained on.
        The saved files are used to resume training if it is interrupted,
        or to evaluate performance of different models on test data without
        having to retrain them from scratch.

        :param self: Used to Represent the instance of the class.
        :param result_save_dir:str: Used to Specify the directory where
        the checkpoint will be saved.
        :param dataset_name:str: Used to Name the dataset.
        :param num_cv:int: Used to Specify the cross validation number.
        :param num_epoch:int: Used to Specify the number of epochs that have
        been completed.
        :param model_net:Type[torch.nn.Module]: Used to Save the model.
        :param data:dict: Used to Save the data, which is a dictionary containing
        the training and validation data.
        :param optimizer:Type[torch.optim.Optimizer]: Used to Save the optimizer
        state.
        :param loss:float: Used to Save the loss value.
        :param override:bool=True: Used to Override the previous checkpoint.
        :param : Used to Save the model.
        :return: None.

        :doc-author: Julian M. Kleber
        """

        self.save_model_checkpoint(
            result_save_dir=result_save_dir,
            dataset_name=dataset_name,
            num_cv=num_cv,
            num_epoch=num_epoch,
            model_net=model_net,
            optimizer=optimizer,
            loss=loss,
            override=override,
        )

        self.save_result(
            result_save_dir=result_save_dir,
            dataset_name=dataset_name,
            data=data,
            num_cv=num_cv,
            num_epoch=num_epoch,
        )
        logging.info(
            f"Successfully saved a checkpoint for epoch {num_epoch} in CV {num_cv}"
        )

    def save_model_checkpoint(
        self,
        result_save_dir: str,
        dataset_name: str,
        num_cv: int,
        num_epoch: int,
        model_net: Type[torch.nn.Module],
        optimizer: Type[torch.optim.Optimizer],
        loss: float,
        override: bool = True,
    ) -> None:
        """
        The save_model function saves the model to a file.

        The save_model function saves the model to a file. The filename is based on
        the dataset name, number of cross-validation folds, and epoch number. The
        file is saved in the result_save_dir directory with an extension of .pt (for
        PyTorch). If this directory does not exist, it will be created before saving
        the file.

        :param result_save_dir:str: Used to specify the directory where the model will be saved.
        :param dataset_name:str: Used to save the model with a name that includes the dataset it was trained on.
        :param num_cv:int: Used to specify which cross validation fold the model is being saved for.
        :param num_epoch:int: Used to save the model at a certain epoch.
        :param model_net:Type[torch.nn.Module]: Used to save the model.
        :param : Used to save the model at a certain frequency.
        :return: None.

        :doc-author: Julian M. Kleber
        """

        save_model_training_checkpoint(
            result_save_dir=result_save_dir,
            dataset_name=dataset_name,
            num_cv=num_cv,
            num_epoch=num_epoch,
            model_net=model_net,
            optimizer=optimizer,
            loss=loss,
            override=override,
        )

    def load_model_checkpoint(
        self,
        checkpoint_path: str,
        model_net: Model,
        optimizer=torch.optim.Optimizer,
    ) -> Model:
        """
        The load_model_checkpoint function loads a model checkpoint from the specified path.

        The function loads a model checkpoint from the specified path, and sets it as the
        model of this evaluation object. The function also returns that loaded model.

        :param self: Used to Refer to the object itself.
        :param checkpoint_path:str: Used to Specify the path to the checkpoint file.
        :param model_net:Model: Used to Specify the model that is being loaded.
        :param optimizer=torch.optim.Optimizer: Used to Load the optimizer state
        from a checkpoint.
        :param : Used to Load the model checkpoint.
        :return: The model.

        :doc-author: Julian M. Kleber
        """

        model_net_cp = load_model_training_checkpoint(
            checkpoint_path=checkpoint_path, model_net=model_net, optimizer=optimizer
        )
        self.model_net = (
            model_net_cp  # set the model of the evaluation object to the checkpoint
        )
        return model_net_cp

    def __str__(self):
        return "Evaluation for " + str(self.model_net) + " with the " + self.name

    def __repr__(self):
        return "Standard Evaluation Object"
