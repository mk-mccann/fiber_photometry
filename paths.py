from os.path import join

<<<<<<< Updated upstream
fp_data_root_dir = r"/home/matt/repos/fiber_photometry/data"
=======
fp_data_root_dir = r"J:\Alja Podgornik\FP_Alja"
>>>>>>> Stashed changes

summary_file = join(fp_data_root_dir, "Multimaze sheet summary.xlsx")

csv_directory = join(fp_data_root_dir, "Multimaze", "Multimaze_Jan21_FPdata_csvs")

behavior_scoring_directory = join(fp_data_root_dir, "Multimaze scoring")

processed_data_directory = join(fp_data_root_dir, "FP_processed data")

modeling_data_directory = join(fp_data_root_dir, "modeling_data")

figure_directory = join(fp_data_root_dir, "plots")
