
class DefaultObject: 
    """
    The default object provides methods every class needs on top of the standard 
    Python functionality. 
    """
    
    def _get_defaults(method_arguments: dict) -> list:
        """
        The _get_defaults function takes a dictionary of arguments and returns a list of values.
        The function checks if the value is None, if it is none then it checks to see if that key exists in the instance variables.
        If so, then it will return the instance variable associated with that key. If not, then nothing happens and None gets returned.

        :param method_arguments:dict: Used to Pass in the arguments that are passed into the method.
        :return: A list of values that are either none or the value provided in the method_arguments dictionary.

        :doc-author: Trelent
        """

        result = []
        instance_variables = vars(self)  # dictionary with instance variables
        instance_attributes = variables.keys()

        for key, value in method_arguemnts.items():
            if key == "self":
                continue

            if value == None and key in instance_attributes:
                result.append(instance_variables[key])
            elif value != None:
                result.append(value)
            elif value is None and key not in instance_attributes:
                result.append(None)
            else:
                result.append(None)
        return result
