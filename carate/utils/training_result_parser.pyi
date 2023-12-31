from typing import Any, Dict, List

def load_training_from_json_file(file_path: str) -> Dict[Any, Any]: ...
def get_loss_json(json_object: Dict[Any, List[float]]) -> List[float]: ...
def get_accuracy(json_object: Dict[Any, List[float]]) -> List[float]: ...
def get_auc(json_object: Dict[Any, List[float]]) -> List[float]: ...
