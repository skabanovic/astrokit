import numpy as np

def bisection(function, pos_1, pos_2, resolution = 1e-6, max_step = 1e6):

    value_1 = function(pos_1)
    value_2 = function(pos_2)

    if (value_1 < resolution) and (value_1 > -resolution):

        return pos_1

    elif (value_2 < resolution) and (value_2 > -resolution):

        return pos_2

    elif np.sign(value_1) != np.sign(value_2):

        mean_pos = (pos_1+pos_2)/2.

        value_1    = function(pos_1)
        mean_value = function(mean_pos)

        step_count = 0

        while step_count < max_step:

            if (mean_value < resolution) and (mean_value > -resolution):

                break

            elif np.sign(value_1) == np.sign(mean_value):

                pos_1 = mean_pos

            elif np.sign(value_1) != np.sign(mean_value):

                pos_2 = mean_pos


            mean_pos = (pos_1+pos_2)/2.

            value_1    = function(pos_1)
            mean_value = function(mean_pos)

            step_count += 1


        if step_count >= max_step :


            print('error: no solution in resonable time found')

        else:

            return mean_pos

    else:

        print("error: no solution in intervall found")
