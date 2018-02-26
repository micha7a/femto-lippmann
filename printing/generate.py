from printing.tools import *
from printing.patterns import *

if True:

    # general parameters
    plot_stuff = True
    speed = 2
    rate = 1
    directory = "patterns/"

    # color pattern parameters
    wlen = 0.521
    layers_ = 20
    axis = "x"

    # height pattern parameters
    line_length = 20

    arr = height_test_array(
        rate=rate,
        line_len=line_length,
        z_steps=layers_,
        speed_limit=speed)

    plot_arr = arr

    # positions, z_steps, delta_z = color_array(plot_stuff, wlen, speed, axis=axis, layers=layers)

    # plot_arr, arr = pattern2array3d_pair(
    #     p0=np.array([0, 0, 100]),
    #     rate=rate,
    #     pattern=positions,
    #     speed=np.ones_like(positions) * speed,
    #     z_steps=z_steps,
    #     delta_z=delta_z)

    if plot_stuff:
        plot_arr = np.array(plot_arr)
    arr = np.array(arr)

    print("full length:", len(arr))

    if plot_stuff:
        print("without markers", len(plot_arr))

    check_array(plot_arr, rate=rate, max_speed=speed + 0.01,
                max_points=3333 * 1000, max_speed_change=speed + 0.01)
    array2file(rate, arr, directory + "color_" + axis + "_" + str(wlen) + "_wlen_" + str(speed) + "_speed.lipp")

    print("file created")

    if plot_stuff:
        plot_pattern(plot_arr)
