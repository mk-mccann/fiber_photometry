from os.path import join


fp_data_root_dir = r"/home/matt/repos/fiber_photometry/data"

summary_file = join(fp_data_root_dir, "Multimaze sheet summary.xlsx")

csv_directory = join(fp_data_root_dir, "Multimaze", "Multimaze_Jan21_FPdata_csvs")

dual_recording_csv_directory = join(fp_data_root_dir, "FP_data_dual_recordings")

behavior_scoring_directory = join(fp_data_root_dir, "Multimaze scoring")

processed_data_directory = join(fp_data_root_dir, "preprocessed_data")

modeling_data_directory = join(fp_data_root_dir, "modeling_data")

figure_directory = join(fp_data_root_dir, "plots")
