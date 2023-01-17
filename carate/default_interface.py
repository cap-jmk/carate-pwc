"""Default Interface defining the Base object for all other objects. 
Especially the __get_default_method is provided for all base objects. 

:author: Julian M. Kleber
"""
from typing import Type, Optional, List, Dict, Any

import logging


logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="train.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
)


class DefaultObject:
    """
    The default object provides methods every class needs on top of the standard
    Python functionality.
    """

    def _get_defaults(self, method_arguments: Dict[Any, Any]) -> List[Any]:
        """
        The _get_defaults function takes a dictionary of arguments and returns a list of values.
        The function checks if the value is None, if it is none then it checks to see if that key exists in the instance variables.
        If so, then it will return the instance variable associated with that key. If not, then nothing happens and None gets returned.

        :param method_arguments:dict: Used to pass in the arguments that are passed into the method.
        :return: A list of values that are either none or the value provided in the method_arguments dictionary.

        :doc-author: Trelent
        """

        result = []
        instance_variables = vars(self)  # dictionary with instance variables
        instance_attributes = instance_variables.keys()

        for key, value in method_arguments.items():
            if key == "self":
                continue
            print(value, instance_variables[key])
            if value is None and key in instance_attributes:
                result.append(instance_variables[key])
            elif value != None:
                result.append(value)
            elif value is None and key not in instance_attributes:
                result.append(None)
            else:
                result.append(None)
        return result
