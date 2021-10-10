class Vector(object):
    def __init__(self, size, is_col=True, fill=0, init_values=None):
        self.v = []
        self.size = size
        self.is_col = is_col

        if init_values is not None:
            l = len(init_values)
            for i in range(size):
                self.v.append(init_values[i % l])
        else:
            self.v = [fill for i in range(size)]

    def __getitem__(self, key):
        return self.v[key]

    def __setitem__(self, key, value):
        self.v[key] = value

    def __str__(self):
        s = '[  '
        lf = "\n" if self.is_col else ""
        for item in self.v:
            s = s + f'{str(item)},  {lf}'
        s += "]"
        return s

    def __check_other(self, other):
        """
        Check validity of self and other.
        If scalar - will broadcast to a vector
        :param other: The other vector or scalar
        :return: same vector or broadcast vector
        """
        if not isinstance(other, Vector):
            if type(other) in [int, float]:
                other = Vector(self.size, True, fill=other)
            else:
                raise ValueError('Wrong parameter type')
        if not self.is_col or not other.is_col:
            raise ValueError('both vectors must be column vectors')
        if self.size != other.size:
            raise ValueError('vectors must be same size')
        return other

    def __add__(self, other):
        """
        Add vectors
        :param other: Other vector or scalar
        :return: The sum of the vectors
        """
        other = self.__check_other(other)
        res = [self.v[i] + other[i] for i in range(self.size)]
        return Vector(self.size, True, init_values=res)

    def __sub__(self, other):
        """
        Subtract vectors
        :param other: Other vector or scalar
        :return: diff between vectors
        """
        other = self.__check_other(other)
        res = [self.v[i] - other[i] for i in range(self.size)]
        return Vector(self.size, True, init_values=res)

    def __rsub__(self, other):
        """
        Subtract vectors when the other is first
        :param other: Other vector or scalar
        :return: diff between vectors
        """
        other = self.__check_other(other)
        res = [other[i] - self.v[i] for i in range(self.size)]
        return Vector(self.size, True, init_values=res)

    def __mul__(self, other):
        """
        Multiply vectors
        :param other: Other vector or scalar
        :return: Multiplication of the vectors
        """
        other = self.__check_other(other)
        res = [self.v[i] * other[i] for i in range(self.size)]
        return Vector(self.size, True, init_values=res)

    def __truediv__(self, other):
        """
        True div vectors
        :param other: Other vector or scalar
        :return: The true div of the vectors
        """
        other = self.__check_other(other)
        res = [self.v[i] / other[i] for i in range(self.size)]
        return Vector(self.size, True, init_values=res)

    def __rtruediv__(self, other):
        """
        True div vectors when the other is first
        :param other: Other vector or scalar
        :return: True div of the vectors
        """
        other = self.__check_other(other)
        res = [other[i] / self.v[i] for i in range(self.size)]
        return Vector(self.size, True, init_values=res)

    def __lt__(self, other):
        other = self.__check_other(other)
        res = [1 if self.v[i] < other[i] else 0 for i in range(self.size)]
        return Vector(self.size, True, init_values=res)

    def __le__(self, other):
        other = self.__check_other(other)
        res = [1 if self.v[i] <= other[i] else 0 for i in range(self.size)]
        return Vector(self.size, True, init_values=res)

    def __eq__(self, other):
        other = self.__check_other(other)
        res = [1 if self.v[i] == other[i] else 0 for i in range(self.size)]
        return Vector(self.size, True, init_values=res)

    def __ne__(self, other):
        other = self.__check_other(other)
        res = [1 if self.v[i] != other[i] else 0 for i in range(self.size)]
        return Vector(self.size, True, init_values=res)

    def __gt__(self, other):
        other = self.__check_other(other)
        res = [1 if self.v[i] > other[i] else 0 for i in range(self.size)]
        return Vector(self.size, True, init_values=res)

    def __ge__(self, w):
        w = self.__check_other(w)
        res = [1 if self.v[i] >= w[i] else 0 for i in range(self.size)]
        return Vector(self.size, True, init_values=res)

    def transpose(self):
        return Vector(self.size, not self.is_col, init_values=self.v)

    def dot(self, other):
        if not self.is_col and other.is_col:
            if self.size == other.size:
                return sum([self.v[i] * other[i] for i in range(self.size)])
            else:
                raise ValueError("Vectors must be same size")
        else:
            raise ValueError("First vector must be row vector and the second must be column vector")

    __radd__ = __add__
    __rmul__ = __mul__
