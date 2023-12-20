from abc import ABC, abstractmethod
from typing import Any, Dict


class Constrainer(ABC):
    """
    Base class for constrainers.

    A constrainer is any kind of class that infers something from a dataset
    and constrains the generation parameters according to that. For instance,
    length constrainers, that automatically infer the length and return maximum
    or minimum number of tokens accordingly.
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_constraints(self) -> Dict[str, Any]:
        """
        Method that return parameters with values to constrain later the
        generation parameters.

        Example:
            output: {"max_tokens": 137, "min_tokens": 32}

        Returns:
            Dict[str, Any]: values to constrain generation parameters.
        """
        ...

    @abstractmethod
    def estimate(self) -> Any:
        """
        Method to estimate values that will be used in `constrain`.

        Returns:
            Any: any kind of value.
        """
        ...

    def constrain(self, generation_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Constrains a generation_config.

        Args:
            generation_config (Dict[str, Any]): a generation config.

        Returns:
            Dict[str, Any]: constrained generation config.
        """
        return {
            **self.get_constraints(),
            **generation_config,
        }
