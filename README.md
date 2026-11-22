# Temperature-Resolved Impedance Analysis of Lithium-Ion Cells

This repository contains the analysis code for the manuscript *Metrological Requirements for Comparable Activation Energies from Temperature-Resolved Impedance Spectroscopy of Commercial Lithium-Ion Cells*.

## 1. Download the UNISA dataset

Download the [UNISA-CAU Thermo-Electrochemical Impedance Spectroscopy Dataset for NCR18650B Li-ion batteries](https://opendata.uni-kiel.de/receive/fdr_mods_00000111). The dataset DOI is [10.57892/100-111](https://doi.org/10.57892/100-111).

## 2. Prepare the data directory

1. Create a `data/` directory in the repository root, alongside `scripts/`, `eislib/`, and `other_data/`.
2. Copy the contents of `other_data/` directly into `data/`, preserving the folder structure. For example, copy `other_data/inhouse/` to `data/inhouse/`.
3. Create `data/unisa_S3/` and extract the downloaded UNISA archive into it. Preserve the original filenames and subfolder structure.

### Final `data/` folder structure

```text
data/
├── inhouse/
│   ├── LGHE2_S1.xlsx
│   ├── all_temp_sony_S1.xlsx
│   ├── Samsung_S2a.xlsx
│   └── Panasonic_S2b.xlsx
├── unisa_S3/
├── LGM50_SoH_S4/
│   └── all_ecm_SoH_temp_S4.xlsx
└── corpora/
    └── impedance_S5.csv
```

The labels **S1, S2a, S2b, S3, S4, and S5** in Table 1 of the manuscript also appear in the corresponding file or folder names. Use these labels to match each dataset to its description in the manuscript.

Before running the analysis, check that the paths in `scripts/00_paths.py` match your local files. The UNISA input path uses `data/unisa_S3/**/EIS_allsoc_*.xlsx` to find the extracted Excel files.

To keep the downloaded UNISA files local, add `/data/unisa_S3/` to the repository's `.gitignore`.

## 3. Set up Python

Open the repository in VS Code, select your Python interpreter, and open **Terminal → New Terminal**. Run the following commands from the repository root.

If the repository provides a `requirements.txt` file, install the listed dependencies.

```bash
python -m pip install -r requirements.txt
```

Otherwise, install the main packages used for analysis, figure generation, and Excel input.

```bash
python -m pip install numpy pandas scipy matplotlib openpyxl
```

Check the imports in any additional scripts you plan to run. Record the Python and package versions used for the analysis so that the environment can be reproduced.

## 4. Run the analysis

To process the UNISA dataset and generate figures, run these commands in order.

```bash
python scripts/01_parse_unisa.py
python scripts/02_fit_unisa.py
python scripts/03_subsampling.py
python scripts/07_figures.py
```

Wait for each script to finish successfully before running the next. Scripts 01–03 prepare inputs used by the figure script. The figure script saves its output in `figs/`. Check the terminal messages for any figures skipped because of missing inputs.

## 5. Cite the data and code

Cite the UNISA dataset separately from the analysis code.

Mattia, L., Beiranvand, H., Liserre, M., and Zamboni, W. (2025). *UNISA-CAU Thermo-Electrochemical Impedance Spectroscopy Dataset for NCR18650B Li-ion batteries*. [https://doi.org/10.57892/100-111](https://doi.org/10.57892/100-111)

The UNISA dataset is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See `LICENSE` for the terms that apply to the repository code.

When citing the analysis code, include the repository URL and the release tag or commit used for the analysis.
