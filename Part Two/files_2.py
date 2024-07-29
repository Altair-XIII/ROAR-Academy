from matplotlib import image
from matplotlib import pyplot
import os

# Read an image file
lenna_path = os.path.dirname(os.path.abspath(__file__))
lenna_filename = lenna_path + '/../samples/' + 'lenna.bmp'
lenna_data = image.imread(lenna_filename)

flag_path = os.path.dirname(os.path.abspath(__file__))
flag_filename = flag_path + '/' + 'japan_flag.png'
flag_data = image.imread(flag_filename)
print(flag_data.shape)

plot_lenna_data = lenna_data.copy()
for width in range(1, flag_data.shape[1]):
    for height in range(1, flag_data.shape[0]):
        plot_lenna_data[height][512 - width] = flag_data[height][flag_data.shape[1] - width] * 255   # Alternatively plot_data[height][width][:] = [255, 0, 0]
# multiply by 255 bc the png file is compressed, using floating point values btw 0 and 1

# Write the modified images
image.imsave(flag_path+'/'+'lenna-flag-mod.jpg', plot_lenna_data)

# use pyplot to plot the image
pyplot.imshow(plot_lenna_data)
pyplot.show()