from vector import Vector


def main():
    print(Vector(4, fill=7))
    print(Vector(5, is_col=False, init_values=[1, 2, 3]))

    try:
        print(Vector(3, fill=1) + Vector(4, fill=2))
    except Exception as err1:
        print('Exception:', err1)
        try:
            print(Vector(3, is_col=False, fill=1) + Vector(3, fill=2))
        except Exception as err2:
            print('Exception', err2)
            print(Vector(3, fill=15) + Vector(3, fill=21))

    x = Vector(3, fill=15)
    x[2] = 7
    print(x[2], '\n')

    v = Vector(3, init_values=[-0.7, -0.03, 1.44])
    vT = v.transpose()
    print(v)
    print(vT, '\n')

    v = Vector(3, init_values=[-0.7, -0.03, 1.44])
    print((15 + (v - 4) * 3) / 2, '\n')

    alpha_W = Vector(3, init_values=[-0.7, -0.03, 1.44])
    dW = Vector(3, init_values=[100, -200, 5])
    same_direction = dW * alpha_W > 0
    alpha_W = alpha_W * same_direction * 1.1 + alpha_W * (1 - same_direction) * -0.5
    print(alpha_W)

    try:
        print(Vector(3, is_col=False, fill=1).dot(Vector(4, fill=2)))
    except ValueError as err1:
        print("Exception:", err1)
        try:
            print(Vector(4, is_col=True, fill=1).dot(Vector(4, fill=2)))
        except ValueError as err2:
            print("Exception:", err2)
            v1, v2 = Vector(3, is_col=False, fill=4), Vector(3, fill=2)
            print(v1.dot(v2), v1, v2, '\n')


if __name__ == '__main__':
    main()
