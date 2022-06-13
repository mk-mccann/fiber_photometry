import functions_io as f_io
from functions_utils import median_of_time_window


if __name__ == "__main__":

    mouse = 3
    day = 4

    data = f_io.load_preprocessed_data(mouse, day)

 #calculating data for cage + experiment 

    # time window is assumed to be in seconds
    # the last two values are the start and end times you want to take the median from
    #median_home_cage = median_of_time_window(data['time'], data['zscore'], 0, 300)
    #median_middle = median_of_time_window(data['time'], data['zscore'], 300, 2100)

# calculating median data for day 3 - shocks

    # time window is assumed to be in seconds
    # the last two values are the start and end times you want to take the median from
    #median_home_cage = median_of_time_window(data['time'], data['zscore'], 0, 300)
    #median_shock = median_of_time_window(data['time'], data['zscore'], 360, 600)
    #median_middle = median_of_time_window(data['time'], data['zscore'], 600, 2640)
    
    # Example for DC recordings
    # median_home_cage_anterior = median_of_time_window(data['time'], data['zscore_anterior'], 0, 300)
    
# Calculating data for 5 min shock segments
    median_home_cage = median_of_time_window(data['time'], data['zscore'], 0, 300)
    median_transfer = median_of_time_window(data['time'], data['zscore'], 300, 360)
    median_shock = median_of_time_window(data['time'], data['zscore'], 360, 600)
    median_5min = median_of_time_window(data['time'], data['zscore'], 600, 900)
    median_10min = median_of_time_window(data['time'], data['zscore'], 900, 1200)
    median_15min = median_of_time_window(data['time'], data['zscore'], 1200, 1500)
    median_20min = median_of_time_window(data['time'], data['zscore'], 1500, 1800)
    median_25min = median_of_time_window(data['time'], data['zscore'], 1800, 2100)
    median_30min = median_of_time_window(data['time'], data['zscore'], 2100, 2400)
    median_35min = median_of_time_window(data['time'], data['zscore'], 2400, 2640)

    print('Median of the home cage period: {}'.format(median_home_cage))
    print('Median of the middle period: {}'.format(median_middle))
