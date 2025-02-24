""" Copyright (c) 2023, ETH Zurich, 
Alexandre Didier*, Robin C. Jacobs*, Jerome Sieber*, Kim P. Wabersich°, Prof. Dr. Melanie N. Zeilinger*, 
*Institute for Dynamic Systems and Control, D-MAVT
°Corporate Research of Robert Bosch GmbH
All rights reserved."""

""" This module defines the abstract class Controller. in addition, it contains the definition of the LinearController
class which inherits from this class """
from abc import abstractmethod


class Controller():
    """ Abstract Class defining Controller objects"""
    @classmethod
    @abstractmethod
    def input(self, x):
        pass


class LinearController(Controller):
    """ This class defines a simple linear controller"""

    def __init__(self, K) -> None:
        super().__init__()
        self.K = K

    def input(self, x):
        """ Returns the next control input given the current state"""
        return self.K @ x

    # def __call__(self, *args, **kwds):
    #    return input(*args, **kwds)
