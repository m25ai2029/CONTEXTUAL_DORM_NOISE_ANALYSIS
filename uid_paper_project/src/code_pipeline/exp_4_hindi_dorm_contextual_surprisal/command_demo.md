
-------------------------------------------------------------------------------
Pipeline for Experiment-4
-------------------------------------------------------------------------------
Sequence of Execution:

train_kneser_ney_lm.py              (for training trigram LM from input huge training CSV file)
                                    (input train file: train_data_demo.csv   
                                    output LM model file: trigram_kneser_ney_model.pkl.gz)

gen_four_variants_context_surp.py   (generate 4 sentence variants from test data)
                                    (input test csv file: test_data_hindi_demo.csv  
                                    output csv: four_variant_demo.csv)

plot_dorm_curves.py                 (for plotting DORM curves for each sentence variants)
                                    (input csv : four_variant_demo.csv
                                    output csv : dorm_plots_demo.png)

simulate_noise.py                   (for simulating single and clustered noise on each sentence variant)
                                    (input csv : four_variant_demo.csv    
                                    output csv : noise_output_demo.csv)

plot_noise_results.py               (for plotting noise results)
                                    (input csv : noise_output_demo.csv)

stats_diagnostics_analysis.py       (for plotting stats and diagnostics)
                                    (input csv : noise_output_demo.csv    
                                    output csv : stats_diag_demo.csv)                                   

t_test_stats.py                     (plot performing paired t-test)
                       

-------------------------------------------------------------------------------
File execution commands
-------------------------------------------------------------------------------

(1)  Call .py file :  train_kneser_ney_lm.py

python3 src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/train_kneser_ney_lm.py \
--input_csv ./data/processed_data/exp_4_demo/train_data_demo.csv \
--output_model ./outputs/mtp_2_features/exp_4_demo_output/trigram_kneser_ney_model.pkl.gz \
--order 3

(2)  Call .py file :  gen_four_variants_context_surp.py

python3 src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/gen_four_variants_context_surp.py \
    --input_csv ./data/processed_data/exp_4_demo/test_data_hindi_demo.csv \
    --output_csv ./outputs/mtp_2_features/exp_4_demo_output/four_variant_demo.csv \
    --order 3

(3)  Call .py file :  plot_dorm_curves.py

python3 src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/plot_dorm_curves.py \
    --input_csv ./outputs/mtp_2_features/exp_4_demo_output/four_variant_demo.csv \
    --output_png ./outputs/mtp_2_features/exp_4_demo_output/hindi_dorm_plots_demo.png

(4)  Call .py file :  simulate_noise.py

python3 src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/simulate_noise.py \
    --input_csv ./outputs/mtp_2_features/exp_4_demo_output/four_variant_demo.csv \
    --output_csv ./outputs/mtp_2_features/exp_4_demo_output/noise_output_demo.csv 

(5)  Call .py file :  plot_noise_results.py

python3 src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/plot_noise_results.py \
    --input_csv ./outputs/mtp_2_features/exp_4_demo_output/noise_output_demo.csv

(6)  Call .py file :  stats_diagnostics_analysis.py

python3 src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/stats_diagnostics_analysis.py \
   --input_csv ./outputs/mtp_2_features/exp_4_demo_output/noise_output_demo.csv \
   --output_summary_csv ./outputs/mtp_2_features/exp_4_demo_output/stats_diag_demo.csv

(7)  Call .py file :  t_test_stats.py

python3 src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/convert_excel_for_stats.py && python3 src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/t_test_stats.py



Additional tasks:
******************************************

preview file noise_output_demo.csv
----------------------------------

python3 src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/preview_sample_file.py \
--input_csv ./outputs/mtp_2_features/exp_4_demo_output/noise_output_demo.csv \
--output_csv ./outputs/mtp_2_features/exp_4_demo_output/sample_noise_output_demo.csv

preview file four_variant_demo.csv
----------------------------------

python3 src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/preview_sample_file.py \
--input_csv ./outputs/mtp_2_features/exp_4_demo_output/four_variant_demo.csv  \
--output_csv ./outputs/mtp_2_features/exp_4_demo_output/sample_four_variant_demo.csv


