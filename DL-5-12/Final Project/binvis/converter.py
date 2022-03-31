from PIL import Image, ImageDraw

from binvis.scurve.hilbert import Hilbert
from binvis.scurve.color import ColorClass


def convert_to_image(size, file_path, name):
    with open(file_path, 'rb') as file:
        data = file.read()

    csource = ColorClass(data)
    map = Hilbert.fromSize(2, size ** 2)
    c = Image.new("RGB", (size, size * 4))
    cd = ImageDraw.Draw(c)
    step = len(csource) / float(len(map) * 4)

    sofar = 0
    for quad in range(4):
        for i, p in enumerate(map):
            off = (i + (quad * size ** 2))
            color = csource.point(
                int(off * step)
            )
            x, y = tuple(p)
            cd.point(
                (x, y + (size * quad)),
                fill=tuple(color)
            )

    c_small = c.resize((150, 450))
    c_small.save(name)

    return c
