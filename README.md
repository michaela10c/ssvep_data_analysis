# SSVEP Data Analysis: CCA, FBCCA, LDA


## Project Overview
This repository contains the data analysis and results for our Steady-State Visually Evoked Potentials (SSVEP) project, which investigates signal preprocessing, feature extraction, and classification for SSVEP signals using Canonical Correlation Analysis (CCA), Filter Bank CCA (FBCCA), and Linear Discriminant Analysis (LDA). This project is built using EEG datasets recorded from two subjects across multiple sessions. The aim is to enhance classification accuracy of brain signals, and potentially improve results through a combination of machine learning and signal processing techniques.

## Folder Structure and File Descriptions

### Data Files
- `subject_1_fvep_led_training_1.mat/.EDF`: EEG data for Subject 1 during the first training session.
- `subject_1_fvep_led_training_2.mat/.EDF`: EEG data for Subject 1 during the second training session.
- `subject_2_fvep_led_training_1.mat/.EDF`: EEG data for Subject 2 during the first training session.
- `subject_2_fvep_led_training_2.mat/.EDF`: EEG data for Subject 2 during the second training session.

### Code Files
- `ssvep_analysis.m`: The MATLAB code used to perform raw and filtered EEG data processing and plot relevant results.
- `ssvep_analysis_full.m`: Comprehensive MATLAB analysis covering all sessions and subjects, including classification results for CCA, FBCCA, and LDA.
- `classInfo_4_5.m`: Class information for the data, with one-hot encoded labels and frequencies corresponding to the tasks.

### Results
- `ssvep_analysis.pdf`: Raw and filtered EEG data plots with trigger information for all channels and epochs.
- `ssvep_analysis_full.pdf`: Full analysis including signal processing, feature extraction, and classification accuracy results.
- `subject_1_fvep_led_training_1_result2d.PNG`: Visualized results for Subject 1 during the first training session via LDA.
- `subject_2_fvep_led_training_1_result2d.PNG`: Visualized results for Subject 2 during the first training session via LDA.

### Paper
- `How_many_people_could_use_an_SSVEP_BCI_simultaneously.pdf`: An exploratory document looking at the simultaneous usability of SSVEP-based BCIs by multiple users.

## How to Use:
1. Load the appropriate .mat or .EDF files for EEG data.
2. Run the provided MATLAB scripts to preprocess, extract features, and classify the data.
3. Visualize the classification results (CCA, FBCCA, LDA) using the included plotting scripts or visual results (PNG files).
4. Refer to the PDF files for detailed visual representations of the data and analysis.

## Slides
This project was done as part of the [2024 BR41N.IO Hackathon](https://www.br41n.io/IEEE-SMC-2024). You can find the **slides** to this project **[here](https://docs.google.com/presentation/d/1HL1KEYquqq7TUbjNczR6fmc6t5TpPdv-/edit?usp=drive_link&ouid=112230274661781285675&rtpof=true&sd=true)**.  
