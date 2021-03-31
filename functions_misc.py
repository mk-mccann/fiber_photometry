import numpy as np


# find CS bool - boolean of CS duration
def find_CS_bool(ts, start_times_CSplus, end_times_CSplus):
    CS_bool = np.zeros(ts.size, dtype=bool)
    preCS_bool = np.zeros(ts.size, dtype=bool)
    postCS_bool = np.zeros(ts.size, dtype=bool)
    for j in np.arange(start_times_CSplus.size):
        start_CS_ind = np.argmin(np.abs(ts - start_times_CSplus[j]))
        end_CS_ind = np.argmin(np.abs(ts - end_times_CSplus[j]))
        CS_bool[start_CS_ind:end_CS_ind] = True
        start_preCS_ind = np.argmin(np.abs(ts - (start_times_CSplus[j] - 30)))
        end_preCS_ind = np.argmin(np.abs(ts - (end_times_CSplus[j] - 30)))
        preCS_bool[start_preCS_ind:end_preCS_ind] = True
        start_postCS_ind = np.argmin(np.abs(ts - (start_times_CSplus[j] + 30)))
        end_postCS_ind = np.argmin(np.abs(ts - (end_times_CSplus[j] + 30)))
        postCS_bool[start_postCS_ind:end_postCS_ind] = True

    return CS_bool, preCS_bool, postCS_bool


def tsplotSlice(corrData, shockStartTimepoints, windowPlusMinus):
    counter = 0
    # tempDf1 = pd.DataFrame()
    tempDf1 = []
    for i in shockStartTimepoints:
        temp1 = corrData[(i - windowPlusMinus): (i + windowPlusMinus)]
        tempDf1.append(temp1)

        counter = counter + 1

    return np.array(tempDf1)
