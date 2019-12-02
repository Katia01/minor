"""
from PIL import Image

color = (0, 255, 0)
im = Image.open('/Users/KatiaSchalk/Desktop/test_vert.jpg')
rgb_im = im.convert('RGB')
print(rgb_im.size()[0])
for x in range(rgb_im.size()[0]):
    for y in range(rgb_im.size()[1]):
        r, g, b = rgb_im.getpixel((x, y))
        if (r,g,b) == colour:
            print(f"Found {colour} at {x},{y}!")
"""
from PIL import Image

im = Image.open('/Users/KatiaSchalk/Desktop/test_vert.jpg')
width, height = im.size
pixels = list(im.getdata())
print(pixels)
index = pixels.index((255,64, 19))
print(index)
