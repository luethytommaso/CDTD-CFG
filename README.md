This code is used for the Master Thesis: "CDTD-CFG: Targeted Out-of-Distribution Synthesis for Tabular Medical Data"

## Contents
The repo consists of two model frameworks (CDTD-CFG and Synthpop), and a main used for evaluation.

It contains a CDTD-CFG framework, that is an extension on CDTD framework by Mueller, M., Gruber, K., & Fok, D. (2023). Continuous diffusion for mixed-type tabular data. arXiv preprint arXiv:2312.10431. Their original code can be found here https://github.com/muellermarkus/cdtd_simple.

The Synthpop framework contains an extension of the Synthpop R package to Python. Nowok, B., Raab, G. M., & Dibben, C. (2016). synthpop: Bespoke creation of synthetic data in R. Journal of statistical software, 74, 1-26.

## Notes
Throughout the CDTD-CFG code, method and class descriptions denote where changes have been made to the model architecture to allow for both the original CDTD outputs, as well as the altered CDTD-CFG outputs (both ID and OOD).

All outputs were attained by running the main.ipynb script in Google Colab, using their T4 GPU.
