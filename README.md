# Experiment 4 — Hindi Contextual Surprisal, DORM and Noise Simulation - Demo Version

**  Execution of all files for macOS and Windows    
**  Project:      M.Tech MTP-2 : Hindi contextual-surprisal experiment   - Demo Run
**  Entry point:  Run commands from the `uid_paper_project` project root  
**  Model:        Order-3 interpolated Kneser-Ney language model  

 ** Scope:  
 This readme file describes: 
  1- Directory structure
  2- Code path
  3- Data path
  4- Output path
  5- Commands to be entered into the terminal window for execution of Python scripts for both MacOs and Windows
  6- List of output files/plots generated
  
  It covers training  of LM model, four-variant generation, DORM plotting, noise simulation, noise plotting, diagnostics, statistics preparation, paired t-testing, and CSV preview.

## 1. Order of execution
 

-------------------------------------------------------------------------------
Pipeline for Experiment-4  - Hindi
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
                       


## 2.  Required directory structure

Create the directory tree below. Folder and script names should match exactly

uid_paper_project/                         -- Project root: launch every command here
|
|-- README_hindi_demo.md                    -- this document
|-- requirements.txt                       -- Python dependencies
|-- src/
|   |-- code_pipeline/
|       |-- exp_4_hindi_dorm_contextual_surprisal/                   -- keep all Python files here
|           |-- train_kneser_ney_lm.py
|           |-- gen_four_variants_context_surp.py
|           |-- plot_dorm_curves.py
|           |-- simulate_noise.py
|           |-- plot_noise_results.py
|           |-- stats_diagnostics_analysis.py
|           |-- convert_excel_for_stats.py
|           |-- t_test_stats.py
|           |-- preview_sample_file.py
|
|-- data/
|   |-- processed_data/
|       |-- exp_4_demo/                 -- Source input file
|           |-- train_data_demo.csv      --  Hindi training data
|           |-- test_data_hindi_demo.csv       --  cleaned 10-token Hindi test data
|
-- outputs/
    |-- mtp_2_features/
        |-- exp_4_demo_output/             -- all outputs generated in this directory
            |-- trigram_kneser_ney_model.pkl.gz
            |-- four_variant_demo.csv
            |-- dorm_plots_demo.png
            |-- noise_output_demo.csv
            |-- stats_diag_demo.csv
            |-- stats_ip_file_demo.csv
            |-- statistical_results.csv
            |-- sample_four_variant_demo.csv   
            |-- sample_noise_output_demo.csv   
            |-- noise_plots/
                |-- bit_loss_single_noise.png
                |-- bit_loss_clustered_noise.png
                |-- catastrophic_loss_comparison.png

Note: source input demo file is large in sie , around 80 MB , so not included as part of submission. However, link to download the train file is given below:

https://drive.google.com/file/d/1OLH4X1Dqaqgp6r7CYB5HRRr822Q6DzzE/view?usp=share_link


## 3. Python installation and virtual environment

Recommended: 
- 64-bit Python 3.11 (minimum Python 3.10). 
- The scripts calls NumPy, pandas, Matplotlib and SciPy external libraries. 

### Running on macOS (Terminal / zsh)

Open Terminal -  The first line creates the project root if it is not already present -  it does not erase an existing project:

```bash
mkdir -p "$HOME/uid_paper_project"
cd "$HOME/uid_paper_project"
python3 --version

# Create once, within the Project root:
python3 -m venv .venv

# Activate for each new terminal session:
source .venv/bin/activate

# Install packages:
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Confirm the active Python and its libraries:
which python
python --version
python -c 'import numpy,pandas,matplotlib,scipy; print("Dependencies OK")'
```

The terminal prompt commonly starts with `(.venv)` after activation. To exit, run `deactivate`.
On a new terminal, activate the same `.venv`.

### Running on Windows 11 (PowerShell)

Open PowerShell - The first command creates the project root if it does not already exist:

```powershell
New-Item -ItemType Directory -Force -Path "$HOME\uid_paper_project" | Out-Null
cd "$HOME\uid_paper_project"
py --version

# Create once, within the Project root:
py -3.11 -m venv .venv

# Activate for each new PowerShell session:
.\.venv\Scripts\Activate.ps1

# Install packages:
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Confirm the active Python and its libraries:
python --version
python -c "import numpy,pandas,matplotlib,scipy; print('Dependencies OK')"
```

If Python 3.11 is not installed but another supported Python 3.10+ is installed, use `py -m venv .venv` instead of `py -3.11 -m venv .venv`.

If Windows PowerShell blocks activation because of its execution policy, either use the venv's Python directly 
(`.\.venv\Scripts\python.exe`) or for that PowerShell session only, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

To exit, run `deactivate`. 
In classic Windows Command Prompt (`cmd.exe`), activation is `.venv\Scripts\activate.bat`, the commands below are written for **PowerShell**. The commands run unchanged with `python` after activation.

### 3.1 Create folders from Terminal / PowerShell

Run these from the project root. You may skip already-existing folders

**macOS:**

```bash
mkdir -p src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal
mkdir -p data/processed_data/exp_4_demo
mkdir -p outputs/mtp_2_features/exp_4_demo_output/noise_plots
```

**Windows PowerShell:**

```powershell
New-Item -ItemType Directory -Force -Path "src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal", "data/processed_data/exp_4_demo", "outputs/mtp_2_features/exp_4_demo_output/noise_plots" | Out-Null
```

Copy the nine Python scripts to `src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/`, and put the CSVs in the specified `data/processed_data/exp_4_demo/` directory. `requirements.txt` and this README belong at the project root.

## 4. Training and test datasets

4.1 Train:
- Input training file - `train_data_demo.csv` located `data/processed_data/exp_4_demo/` 
- Builds vocabulary and trains the Kneser–Ney trigram LM
- output LM model file is `trigram_kneser_ney_model.pkl.gz` located in `exp_4_demo_output/trigram_kneser_ney_model.pkl.gz`
4.2 Test:
- `test_data_hindi_demo.csv` located in `data/processed_data/exp_4_demo/`  
- Generates the four variants and evaluates contextual surprisal 
- It gives output CSV with `sent_count`, `sentence` columns; exactly 10 cleaned tokens per sentence
- output file located in `exp_4_demo_output/four_variant_demo.csv `


## 5. Demo experiment: commands for each script


### STEP 1 — Train the trigram Kneser–Ney model

Script: `src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/train_kneser_ney_lm.py`  
Input : `data/processed_data/exp_4_demo/train_data_demo.csv`  
Output: `outputs/mtp_2_features/exp_4_demo_output/trigram_kneser_ney_model.pkl.gz`

The training code reads the full training CSV file: it builds word counts and vocabulary first, then builds n-gram counts, with a fixed discount of 0.75 and a rare-word threshold of 1. It saves the trained model as a gzipped Python pickle. It reports progress during training and accepts `--input_csv`, `--output_model` and `--order`.

**macOS**
```bash
python src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/train_kneser_ney_lm.py \
  --input_csv ./data/processed_data/exp_4_demo/train_data_demo.csv \
  --output_model ./outputs/mtp_2_features/exp_4_demo_output/trigram_kneser_ney_model.pkl.gz \
  --order 3
```

**Windows PowerShell**
```powershell
python src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/train_kneser_ney_lm.py --input_csv ./data/processed_data/exp_4_demo/train_data_demo.csv --output_model ./outputs/mtp_2_features/exp_4_demo_output/trigram_kneser_ney_model.pkl.gz --order 3
```


### STEP 2 — Generate Actual, Uniform, Asymmetric and Random variants

Script: `src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/gen_four_variants_context_surp.py`  
Input : `data/processed_data/exp_4_demo/test_data_hindi_demo.csv` and the trained LM model
Output: `outputs/mtp_2_features/exp_4_demo_output/four_variant_demo.csv`

The script loads the trained model from a fixed path inside the code, chosen by `--order`. For `--order 3`, that path is `./outputs/mtp_2_features/exp_4_demo_output/trigram_kneser_ney_model.pkl.gz`. It scores the actual sentence, optimizes a Uniform order by retaining swaps that lower DORM, generates Asymmetric order by reducing information centre, and generates a seeded Random order. Contextual surprisal is recalculated after reordering. Each successful test sentence creates four output rows.

The output columns in the uploaded version are:

```text
sent_count, type, sentence, DORM, token_ic_json, total_bits
```

`token_ic_json` contains token-wise surprisal and diagnostic records. 

**macOS**
```bash
python src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/gen_four_variants_context_surp.py \
  --input_csv ./data/processed_data/exp_4_demo/test_data_hindi_demo.csv \
  --output_csv ./outputs/mtp_2_features/exp_4_demo_output/four_variant_demo.csv \
  --order 3
```

**Windows PowerShell**
```powershell
python src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/gen_four_variants_context_surp.py --input_csv ./data/processed_data/exp_4_demo/test_data_hindi_demo.csv --output_csv ./outputs/mtp_2_features/exp_4_demo_output/four_variant_demo.csv --order 3
```


### STEP 3 — Plot the DORM distributions

Script: `src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/plot_dorm_curves.py`  
Input : `outputs/mtp_2_features/exp_4_demo_output/four_variant_demo.csv`  
Output: `outputs/mtp_2_features/exp_4_demo_output/dorm_plots_demo.png`

This script reads `type` and `DORM` from the four-version dataset and draws a raincloud-style distribution of sqrt(DORM), with mean and median indicators. Output PNG filename is `dorm_plots_demo.png`

**macOS**
```bash
python src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/plot_dorm_curves.py \
  --input_csv ./outputs/mtp_2_features/exp_4_demo_output/four_variant_demo.csv \
  --output_png ./outputs/mtp_2_features/exp_4_demo_output/dorm_plots_demo.png
```

**Windows PowerShell**
```powershell
python src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/plot_dorm_curves.py --input_csv ./outputs/mtp_2_features/exp_4_demo_output/four_variant_demo.csv --output_png ./outputs/mtp_2_features/exp_4_demo_output/dorm_plots_demo.png
```


### STEP 4 — Simulate single and clustered deletion noise

Script: `src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/simulate_noise.py`  
Input : `outputs/mtp_2_features/exp_4_demo_output/four_variant_demo.csv`  
Output: `outputs/mtp_2_features/exp_4_demo_output/noise_output_demo.csv`

For 10-token sentences, the script deletes 3 tokens. In single noise, 3 random positions are chosen, in clustered noise,  consecutive 3-position window is chosen. The same selected deletion positions are applied to all four versions belonging to one `sent_count` group. Deleted bits, proportion lost, and a catastrophic-loss flag (at least 50% of original model-assigned bits) are recorded. Noise is simulated on the already-scored tokens.

This code reads input in 50,000-row chunks, carries unfinished sentence groups between chunks, and uses seed 42.

**macOS**
```bash
python src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/simulate_noise.py \
  --input_csv ./outputs/mtp_2_features/exp_4_demo_output/four_variant_demo.csv \
  --output_csv ./outputs/mtp_2_features/exp_4_demo_output/noise_output_demo.csv
```

**Windows PowerShell**
```powershell
python src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/simulate_noise.py --input_csv ./outputs/mtp_2_features/exp_4_demo_output/four_variant_demo.csv --output_csv ./outputs/mtp_2_features/exp_4_demo_output/noise_output_demo.csv
```


### STEP 5 — Generate three noise-result figures

Script          : `src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/plot_noise_results.py`  
Input           : `outputs/mtp_2_features/exp_4_demo_output/noise_output_demo.csv`  
Output directory: `outputs/mtp_2_features/exp_4_demo_output/noise_plots/`

The script produces these three exact files:

```text
noise_plots/bit_loss_single_noise.png
noise_plots/bit_loss_clustered_noise.png
noise_plots/catastrophic_loss_comparison.png
```

**macOS**
```bash
python src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/plot_noise_results.py \
  --input_csv ./outputs/mtp_2_features/exp_4_demo_output/noise_output_demo.csv
```

**Windows PowerShell**
```powershell
python src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/plot_noise_results.py --input_csv ./outputs/mtp_2_features/exp_4_demo_output/noise_output_demo.csv
```

### STEP 6 — Produce contextual-surprisal and noise diagnostics

Script: `src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/stats_diagnostics_analysis.py`  
Input : `outputs/mtp_2_features/exp_4_demo_output/noise_output_demo.csv`  
Output: `outputs/mtp_2_features/exp_4_demo_output/stats_diag_demo.csv`

This script calculates per-version aggregate statistics such as average total bits, average surprisal, maximum single-token share, top-three-token share, maximum consecutive-three-token share, information centre, single/clustered loss, catastrophic-loss proportions, and diagnostic counts for the language-model scoring levels.


**macOS**
```bash
python src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/stats_diagnostics_analysis.py \
  --input_csv ./outputs/mtp_2_features/exp_4_demo_output/noise_output_demo.csv \
  --output_summary_csv ./outputs/mtp_2_features/exp_4_demo_output/stats_diag_demo.csv
```

**Windows PowerShell**
```powershell
python src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/stats_diagnostics_analysis.py --input_csv ./outputs/mtp_2_features/exp_4_demo_output/noise_output_demo.csv --output_summary_csv ./outputs/mtp_2_features/exp_4_demo_output/stats_diag_demo.csv
```


### STEP 7 — Prepare the reduced statistical-input CSV

Script              : `src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/convert_excel_for_stats.py`  
Input (hard-coded)  : `outputs/mtp_2_features/exp_4_demo_output/noise_output_demo.csv`  
Output (hard-coded) : `outputs/mtp_2_features/exp_4_demo_output/stats_ip_file_demo.csv`

This script reads a CSV file, removes bulky token/deletion metadata columns, and saves another CSV.

**macOS**
```bash
python src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/convert_excel_for_stats.py
```

**Windows PowerShell**
```powershell
python src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/convert_excel_for_stats.py
```

### STEP 8 — Perform paired DORM t-tests

Script              : `src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/t_test_stats.py`  
Input (hard-coded)  : `outputs/mtp_2_features/exp_4_demo_output/stats_ip_file_demo.csv`  
Output (hard-coded) : `outputs/mtp_2_features/exp_4_demo_output/statistical_results.csv`

The code pivots DORM into one row per `sent_count` and one column per version, then runs five paired comparisons using `scipy.stats.ttest_rel`, also reporting the absolute paired effect size (`|d_z|`). It reads the prepared `stats_ip_file_demo.csv` created by Step 7.

**macOS**
```bash
python src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/t_test_stats.py
```

**Windows PowerShell**
```powershell
python src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/t_test_stats.py
```

### STEP 9 — Optional CSV previews (500 data rows each)

Script: `src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/preview_sample_file.py`  
Input : the existing full CSV file  
Output: a small CSV file containing the header and up to 500 data rows.

**macOS — four-version preview**
```bash
python src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/preview_sample_file.py \
  --input_csv ./outputs/mtp_2_features/exp_4_demo_output/four_variant_demo.csv \
  --output_csv ./outputs/mtp_2_features/exp_4_demo_output/sample_four_variant_demo.csv
```

**macOS — noise preview**
```bash
python src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/preview_sample_file.py \
  --input_csv ./outputs/mtp_2_features/exp_4_demo_output/noise_output_demo.csv \
  --output_csv ./outputs/mtp_2_features/exp_4_demo_output/sample_noise_output_demo.csv
```

**Windows PowerShell — four-version preview**
```powershell
python src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/preview_sample_file.py --input_csv ./outputs/mtp_2_features/exp_4_demo_output/four_variant_demo.csv --output_csv ./outputs/mtp_2_features/exp_4_demo_output/sample_four_variant_demo.csv
```

**Windows PowerShell — noise preview**
```powershell
python src/code_pipeline/exp_4_hindi_dorm_contextual_surprisal/preview_sample_file.py --input_csv ./outputs/mtp_2_features/exp_4_demo_output/noise_output_demo.csv --output_csv ./outputs/mtp_2_features/exp_4_demo_output/sample_noise_output_demo.csv
```

## 6. Output generated files from running above experiments 


 `trigram_kneser_ney_model.pkl.gz` 
`four_variant_demo.csv` 
`dorm_plots.png` 
`noise_output_demo.csv` 
`noise_plots/bit_loss_single_noise.png` 
`noise_plots/bit_loss_clustered_noise.png` 
`noise_plots/catastrophic_loss_comparison.png` 
`stats_diag_demo.csv` 
`stats_ip_file_demo.csv` 
`statistical_results.csv` 
`sample_four_variant_demo.csv`, `sample_noise_output_demo.csv` 

- all paths of above files are within `outputs/mtp_2_features/exp_4_demo_output/`.
