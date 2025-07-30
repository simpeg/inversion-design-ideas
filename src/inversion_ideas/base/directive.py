"""
Base class for directives.
"""
from abc import ABC, abstractmethod


class Directive(ABC):
    """
    Abstract class for directives.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def initialize(self):
        """
        Initialize the directive.
        """

    @abstractmethod
    def __call__(self):
        pass
