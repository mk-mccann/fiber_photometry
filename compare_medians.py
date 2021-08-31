import functions_io as f_io
from functions_utils import median_of_time_window


if __name__ == "__main__":

    mouse = 2
    day = 1

    data = f_io.load_preprocessed_data(mouse, day)

    # time window is assumed to be in seconds
    # the last two values are the start and end times you want to take the median from
    median_home_cage = median_of_time_window(data['time'], data['zscore'], 0, 300)
    median_middle = median_of_time_window(data['time'], data['zscore'], 1000, 1300)
    
    # Example for DC recordings
    # median_home_cage_anterior = median_of_time_window(data['time'], data['zscore_anterior'], 0, 300)
    

    print('Median of the home cage period: {}'.format(median_home_cage))
    print('Median of the middle period: {}'.format(median_middle))
