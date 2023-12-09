import collections
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

import numpy as np
import logging


logger = logging.getLogger(__name__)


class HyperParam(ABC):
    """
    Represent an hyper-parameter
    """
    def __init__(self, name: str, default_value: Any, current_value: Any = None):
        """

        Args:
            name: a unique name of an hyper-parameter
            default_value: the default value of an hyper-parameter
            current_value: the current value of an hyper-parameter
        """
        self.default_value = default_value
        self.name = name

        if current_value is None:
            self.current_value = default_value
        else:
            self.current_value = current_value

    def set_value(self, value: Any) -> None:
        """
        Set the current value of an hyper-parameter

        Args:
            value: the new current value
        """
        self.current_value = value

    def get_value(self) -> Any:
        """
        return the current value of an hyper-parameter
        """
        return self.current_value

    @abstractmethod
    def randomize(self) -> None:
        """
        Randomize the current value of an hyper-parameter
        """
        pass


class DiscreteValue(HyperParam):
    """
    Discrete value. This can be useful to select one choice among many
    """
    def __init__(self, name: str, default_value: Any, values: List[Any]):
        assert isinstance(values, list)
        assert default_value in values, f'invalid default! name={name} default_value={default_value} ' \
                                        f'not in values. Choices={values}'
        self.values = values
        super().__init__(name, default_value=default_value)

    def set_value(self, value: Any) -> None:
        assert value in self.values
        super().set_value(value)

    def randomize(self):
        v = np.random.randint(low=0, high=len(self.values))
        self.set_value(self.values[v])

    def __repr__(self):
        return f'DiscreteValue value={self.current_value}'


def create_discrete_value(name: str, default_value: Any, values: List[Any]) -> Any:
    p = DiscreteValue(name, default_value, values)
    v = register_hparam(p)
    logger.debug(f'param={name}, value={v}')
    return v


class DiscreteMapping(HyperParam):
    """
    Map discrete value to another discrete value
    
    e.g., this can be useful to test activation function as hyper-parameter
    """
    def __init__(self, name: str, default_value: Any, mapping: Dict[Any, Any]):
        """

        Args:
            name: the name of the hyper-parameter. Must be unique
            default_value: a default value (`key` in kvp)
            mapping: a mapping of key to value
        """
        assert isinstance(mapping, dict)
        assert default_value in mapping, f'`default_value={default_value}` is not in kvp! ' \
                                         f'Must be one of {mapping.keys()}'
        self.mapping = mapping
        self.key_value_list = list(mapping.items())
        super().__init__(name=name, default_value=default_value)

    def set_value(self, value: Any):
        assert value in self.mapping, f'current_value must be in the dictionary! c={value}'
        self.current_value = value

    def get_value(self):
        return self.mapping[self.current_value]

    def randomize(self):
        index = np.random.randint(low=0, high=len(self.key_value_list))
        key, _ = self.key_value_list[index]
        self.set_value(key)

    def __repr__(self):
        return f'DiscreteMapping value={self.current_value}'


def create_discrete_mapping(name: str, default_value: Any, mapping: Dict[Any, Any]) -> Any:
    p = DiscreteMapping(name, default_value, mapping)
    v = register_hparam(p)
    logger.debug(f'param={name}, value={v}')
    return v


class DiscreteInteger(HyperParam):
    """
    Represent an integer hyper-parameter
    """
    def __init__(self, name: str, default_value: int, min_range: int, max_range: int):
        """

        Args:
            name: the name of the hyper-parameter. Must be unique
            default_value: the default value
            min_range: maximum integer (inclusive) to be generated
            max_range: minimum integer (inclusive) to be generated
        """
        assert max_range >= min_range
        self.max_range = max_range
        self.min_range = min_range
        super().__init__(name, default_value)

    def randomize(self) -> None:
        v = np.random.randint(low=self.min_range, high=self.max_range + 1)
        self.set_value(v)

    def __repr__(self):
        return f'DiscreteInteger value={self.current_value} min={self.min_range}, max={self.max_range}'


def create_discrete_integer(name: str, default_value: Any, min_range: int, max_range: int) -> Any:
    p = DiscreteInteger(name, default_value, min_range, max_range)
    v = register_hparam(p)
    logger.debug(f'param={name}, value={v}')
    return v


class DiscreteBoolean(HyperParam):
    """
    Represent a boolean hyper-parameter
    """
    def __init__(self, name, default_value):
        """

        Args:
            name: the name of the hyper-parameter. Must be unique
            default_value: the initial boolean value
        """
        assert isinstance(default_value, bool)
        super().__init__(name, default_value=default_value)

    def randomize(self) -> None:
        v = bool(np.random.randint(low=0, high=1 + 1))
        self.set_value(v)

    def __repr__(self):
        return f'DiscreteBoolean value={self.current_value}'


def create_boolean(name: str, default_value: Any) -> bool:
    p = DiscreteBoolean(name, default_value)
    v = register_hparam(p)
    logger.debug(f'param={name}, value={v}')
    return v


class ContinuousUniform(HyperParam):
    """
    Represent a continuous hyper-parameter
    """
    def __init__(self, name: str, default_value: float, min_range: float, max_range: float):
        """

        Args:
            name: the name of the hyper-parameter. Must be unique
            default_value:
            min_range: minimum (inclusive) to be generated
            max_range: maximum (inclusive) to be generated
        """
        assert max_range >= min_range
        self.max_range = max_range
        self.min_range = min_range
        super().__init__(name, default_value)

    def randomize(self) -> None:
        v = np.random.uniform(low=self.min_range, high=self.max_range)
        self.set_value(v)

    def __repr__(self):
        return f'ContinuousUniform value={self.current_value}, min={self.min_range}, max={self.max_range}'


def create_continuous_uniform(name: str, default_value: float, min_range: float, max_range: float) -> float:
    p = ContinuousUniform(name, default_value, min_range, max_range)
    v = register_hparam(p)
    logger.debug(f'param={name}, value={v}')
    return v

    
class ContinuousPower(HyperParam):
    """
    Represent a continuous power hyper-parameter
    
    This type of distribution can be useful to test e.g., learning rate hyper-parameter. Given a
    random number x generated from uniform interval (min_range, max_range), return 10 ** x

    Examples:
        >>> hp1 = ContinuousPower('hp1', default_value=0.1, exponent_min=-5, exponent_max=-1)
        ``hp1.get_value()`` would return a value in the range(1e-1, 1e-5)
    """
    def __init__(self, name: str, default_value: float, exponent_min: float, exponent_max: float):
        """
        Args:
            name: the name of the hyper-parameter. Must be unique
            default_value: the current value of the parameter (power will ``NOT`` be applied)
            exponent_min: minimum floating number (inclusive) of the power exponent to be generated
            exponent_max: max_range: max floating number (inclusive) of the power exponent to be generated
        """
        assert exponent_max >= exponent_min
        assert default_value >= 10 ** exponent_min, 'make sure the current value must have the power already ' \
                                                    'applied and be within the generated interval'
        assert default_value <= 10 ** exponent_max, 'make sure the current value must have the power already ' \
                                                    'applied and be within the generated interval'
        self.exponent_max = exponent_max
        self.exponent_min = exponent_min
        super(ContinuousPower, self).__init__(name, default_value)

    def randomize(self) -> None:
        uniform = np.random.uniform(low=self.exponent_min, high=self.exponent_max)
        v = 10 ** uniform
        self.set_value(v)

    def __repr__(self):
        return f'ContinuousPower value={self.current_value}, min={self.exponent_min}, max={self.exponent_max}'


def create_continuous_power(name: str, default_value: float, exponent_min: float, exponent_max: float) -> float:
    p = ContinuousPower(name, default_value, exponent_min, exponent_max)
    v = register_hparam(p)
    logger.debug(f'param={name}, value={v}')
    return v


class HyperParameters:
    """
    Holds a repository a set of hyper-parameters
    """
    def __init__(self,
                 hparams: Optional[Dict[str, HyperParam]] = None,
                 randomize_at_creation: bool = False,
                 hparams_to_randomize: Optional[List[str]] = None):
        """
        Create the hyper-parameter repository

        Args:
            hparams: pre-existing hyper-parameters or None
            randomize_at_creation: if True, the hyper-parameter will not have
                take default value at creation but random
            hparams_to_randomize: this is the list og hyper-parameters to randomize. Other hyper-parameters
                will be kept constant during optimization. If `None`, all hyper-parameters will be
                randomized. This can be a regular expression (e.g., `trw.optimizer.*` so that we can match hierarchy
                of hyper-parameters)
        """
        self.hparams_to_randomize = hparams_to_randomize
        self.randomize_at_creation = randomize_at_creation
        if hparams is not None:
            self.hparams = hparams
        else:
            self.hparams = collections.OrderedDict()

    def hparam_to_be_randomized(self, haparam_name: str) -> bool:
        if self.hparams_to_randomize is None:
            return True
        for pattern in self.hparams_to_randomize:
            if re.search(pattern, haparam_name) is not None:
                return True
        return False

    def create(self, hparam: HyperParam) -> Any:
        """
        Create an hyper parameter if it is not already present. If it is present,
        the given `hparam` is ignored

        Args:
            hparam: the hyper-parameter

        Returns:
            the hyper parameter value
        """
        assert isinstance(hparam, HyperParam), 'must be an instance of HyperParam'

        stored_hparam = self.hparams.get(hparam.name)
        if stored_hparam is None:
            if self.randomize_at_creation:
                if self.hparam_to_be_randomized(hparam.name):
                    # parameters not to be optimized should be
                    # created with their default value!
                    hparam.randomize()
            self.hparams[hparam.name] = hparam
            stored_hparam = hparam
        else:
            assert type(stored_hparam) == type(hparam), f'type mismatch (got={type(stored_hparam)} vs ' \
                                                        f'created={type(hparam)}. Hyper-parameter name collision?)! '

        assert stored_hparam is not None
        return stored_hparam.get_value()

    def randomize(self) -> None:
        """
        Set hyper-parameter to a random value
        """
        for name, hparam in self.hparams.items():
            if self.hparam_to_be_randomized(name):
                hparam.randomize()

    def __getitem__(self, name: str):
        hp = self.hparams.get(name)
        assert hp is not None, f'`{name}` is not defined in the hyper-parameters!'
        return hp

    def get_value(self, name: str) -> Any:
        """
        Return the current value of an hyper-parameter
        """
        hparam = self.hparams.get(name)
        assert hparam is not None, 'can\'t find hparam=%s' % hparam
        return hparam.get_value()

    def __str__(self):
        return str(self.hparams)

    def __repr__(self):
        return str(self.hparams)

    def __len__(self):
        return len(self.hparams)


class HyperParameterRepository:
    """
    Holds the current hyper-parameters
    """
    current_hparams: HyperParameters = HyperParameters()

    @staticmethod
    def reset(new_hparams: Optional[HyperParameters] = None) -> None:
        """
        Replace the existing hyper parameters by a new one

        Args:
            new_hparams: the new hyper-parameters
        """
        if new_hparams is None:
            new_hparams = HyperParameters()

        HyperParameterRepository.current_hparams = new_hparams


def register_hparam(hparam: HyperParam) -> Any:
    """
    Create a hyper-parameter and record it in the :class:`HyperParameterRepository` repository

    Args:
        hparam: the hyper-parameter to be created

    Returns:
        the value of the hyper-parameter
    """
    return HyperParameterRepository.current_hparams.create(hparam)
