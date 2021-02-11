import matplotlib.pyplot as plt
import numpy as np
import random
import unit10.b_utils as u10


class MyClass:
    static_v = 5

    def __init__(self, y, x=9):
        self.__x = x
        self.y = y

    def sum(self, const):
        return self.__x + self.y + const

    @staticmethod
    def static_f():
        print("static example")

    def __str__(self):
        return f'x={self.__x}, y={self.y}'
