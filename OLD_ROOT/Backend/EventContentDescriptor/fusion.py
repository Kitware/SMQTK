import numpy as np

def geometric_mean(floats):
    """
    Compute geometric mean
    @param floats: sequence of floats
    @return: float
    """

    temp1 = np.array(floats)
    return np.exp(np.mean(np.log(temp1)))


def fusion_geometric_2col_dict(input_list):
    """
    Conduct geometric fusion on list of inputs
    @param input_list: each input in the list is 2-column dictionary with clipid as key, score as value
    @return: a 2 column dictionary
    """

    temp = input[0].copy()

    if len(input_list) > 1:
        for input in input_list[1:]:
            for (key, value) in input.items():
                try:
                    temp[key].append(value)
                except:
                    temp[key] = [value]

    output = dict()
    for key in temp.iterkeys():
        output[key] = geometric_mean(temp[key])

    del temp
    return output

