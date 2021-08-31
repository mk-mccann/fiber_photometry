from os.path import join


fp_data_root_dir = r"J:\Alja Podgornik\FP_Alja\January '21 WILD TYPE\Multimaze"

csv_directory = join(fp_data_root_dir, "raw_data")

behavior_scoring_directory = join(fp_data_root_dir, "behavior_scoring")

processed_data_directory = join(fp_data_root_dir, "preprocessed_data")

modeling_data_directory = join(fp_data_root_dir, "modeling_data")

figure_directory = join(fp_data_root_dir, "figures")


summary_file = join(fp_data_root_dir, "Multimaze sheet summary.xlsx")
dual_recording_csv_directory = join(fp_data_root_dir, "FP_data_dual_recordings")