%% Data Preprocessing

% Load session data
data1_session1 = load('subject_1_fvep_led_training_1.mat');

% Extract parts 
sample_time = data1_session1.y(1, 1001:end); % Time samples from the 1001st sample onward
EEG_data = data1_session1.y(2:9, 1001:end); % EEG data from CH2-9, starting from the 1001st sample
trigger_info = data1_session1.y(10, 1001:end); % Trigger information from 1001st sample
lda_output = data1_session1.y(11, 1001:end); % LDA classification output from 1001st sample

% Sampling rate
fs = 256; % Sampling rate in Hz

% Bandpass filter settings (5-30 Hz) to better capture SSVEP frequencies
[b_bandpass, a_bandpass] = butter(4, [5 30]/(fs/2), 'bandpass'); % 4th-order Butterworth bandpass filter

% Notch filter settings at 50 Hz
[b_notch, a_notch] = butter(2, [49 51]/(fs/2), 'stop'); % 2nd-order Butterworth notch filter

% Apply the filters to each EEG channel
EEG_data_filtered = zeros(size(EEG_data)); % Initialize matrix for filtered data
for ch = 1:8
    % Apply bandpass filter
    EEG_data_bandpassed = filtfilt(b_bandpass, a_bandpass, EEG_data(ch, :));
    % Apply notch filter at 50 Hz
    EEG_data_filtered(ch, :) = filtfilt(b_notch, a_notch, EEG_data_bandpassed); % Apply notch filter to the bandpassed data
end

% Segmenting the data based on trigger information (Epoching)
epochs = {}; % To store segmented trials
window_size = fs * 3; % 3-second window
onsets = find(trigger_info == 1); % Find where the stimulus is ON (trigger == 1)

% Extract epochs based on the stimulus onsets, ensure all epochs are the same size
max_epoch_length = window_size;
for i = 1:length(onsets)
    if (onsets(i) + window_size - 1) <= length(sample_time)
        epoch = EEG_data_filtered(:, onsets(i):onsets(i) + window_size - 1);
        % Pad or truncate to match max_epoch_length
        epoch = epoch(:, 1:max_epoch_length); % Force all epochs to be the same size
        epochs{i} = epoch;
    end
end

% Number of epochs
num_epochs = length(epochs);  % Ensure num_epochs is defined here

% Frequencies of interest for SSVEP stimuli
frequencies_of_interest = [9, 10, 12, 15]; % Define target frequencies
harmonics = 3; % Increased to 3 harmonics to capture more signal detail
num_channels = size(EEG_data_filtered, 1);

% Ground truth frequencies (from LDA output)
ground_truth_frequencies = lda_output(1:num_epochs); % LDA output as ground truth

%% CCA and FBCCA Analysis (with Parallelism)

acc_cca = zeros(num_epochs, 1);
acc_fbcca = zeros(num_epochs, 1);

parpool; % Start parallel pool

% CCA and FBCCA analysis using parallel processing
parfor i = 1:num_epochs
    epoch = epochs{i}; % Get the i-th trial/epoch

    % --- Check for low variance in epoch ---
    if var(epoch(:)) < 1e-10
        warning(['Epoch ', num2str(i), ' has very low variance and will be skipped.']);
        continue;
    end

    % --- Apply PCA to reduce dimensionality ---
    [coeff, score, ~] = pca(epoch'); % PCA to reduce dimensionality of EEG data
    reduced_epoch = score(:, 1:min(size(score, 2), 4)); % Keep top 4 components

    % --- CCA with Regularization ---
    max_corr = zeros(1, length(frequencies_of_interest));
    for f_idx = 1:length(frequencies_of_interest)
        ref_signals = create_reference_signals(frequencies_of_interest(f_idx), harmonics, size(reduced_epoch, 1), fs); % Ensure ref_signals has the same number of rows as reduced_epoch

        % Regularized CCA
        [A, B, r] = canoncorr(reduced_epoch, ref_signals'); % Transpose ref_signals to match reduced_epoch

        max_corr(f_idx) = max(r); % Take the maximum correlation
    end
    [~, predicted_freq_idx_cca] = max(max_corr); % Predicted frequency (CCA)
    predicted_frequency_cca = frequencies_of_interest(predicted_freq_idx_cca); % Get the predicted frequency

    % Map the predicted frequency to ground truth labels (0 or 3)
    if predicted_frequency_cca == 9 || predicted_frequency_cca == 10
        predicted_label_cca = 0; % Frequencies 9Hz or 10Hz -> Label 0
    elseif predicted_frequency_cca == 12 || predicted_frequency_cca == 15
        predicted_label_cca = 3; % Frequencies 12Hz or 15Hz -> Label 3
    end

    % Check if prediction is correct
    if predicted_label_cca == ground_truth_frequencies(i)
        acc_cca(i) = 1; % Correct prediction
    else
        acc_cca(i) = 0; % Incorrect prediction
    end

    % --- FBCCA (Filter Bank CCA) ---
    num_bands = 7; % Increased to 7 bands for finer frequency resolution
    fb_corr = zeros(1, length(frequencies_of_interest));
    for band_idx = 1:num_bands
        % Create bandpass filters for each band (reduce the filter order to 1)
        f_lower = frequencies_of_interest - 2 * band_idx; % Lower edge
        f_upper = frequencies_of_interest + 2 * band_idx; % Upper edge
        [b_fb, a_fb] = butter(1, [f_lower(f_idx) f_upper(f_idx)]/(fs/2), 'bandpass'); % Filter order reduced to 1

        % Skip the epoch if it's too short for the filter or if it's low variance
        if size(epoch, 2) < 3 * length(b_fb) || var(epoch(:)) < 1e-10
            continue;
        end

        epoch_fb = filtfilt(b_fb, a_fb, epoch); % Apply filter to the padded epoch

        % CCA with filter bank (FBCCA) and regularization
        for f_idx = 1:length(frequencies_of_interest)
            ref_signals = create_reference_signals(frequencies_of_interest(f_idx), harmonics, size(epoch_fb, 2), fs);
            [~, ~, r_fb] = canoncorr(epoch_fb', ref_signals'); % CCA between filtered EEG and reference signals
            fb_corr(f_idx) = max(r_fb) + fb_corr(f_idx); % Accumulate correlation over bands
        end
    end
    [~, predicted_freq_idx_fbcca] = max(fb_corr); % Predicted frequency (FBCCA)
    predicted_frequency_fbcca = frequencies_of_interest(predicted_freq_idx_fbcca); % Get the predicted frequency

    % Map the predicted frequency to ground truth labels (0 or 3)
    if predicted_frequency_fbcca == 9 || predicted_frequency_fbcca == 10
        predicted_label_fbcca = 0; % Frequencies 9Hz or 10Hz -> Label 0
    elseif predicted_frequency_fbcca == 12 || predicted_frequency_fbcca == 15
        predicted_label_fbcca = 3; % Frequencies 12Hz or 15Hz -> Label 3
    end

    % Check if prediction is correct
    if predicted_label_fbcca == ground_truth_frequencies(i)
        acc_fbcca(i) = 1; % Correct prediction
    else
        acc_fbcca(i) = 0; % Incorrect prediction
    end

    % Display progress every 10 epochs
    if mod(i, 10) == 0
        fprintf('Epoch %d of %d processed.\n', i, num_epochs);
    end
end

% Shut down the parallel pool after the parfor loop finishes
delete(gcp); % Programmatically shut down parallel pool

function ref_signals = create_reference_signals(frequency, harmonics, num_samples, fs)
    % Function to create reference signals for CCA analysis
    % frequency: target frequency in Hz
    % harmonics: number of harmonics to include
    % num_samples: number of time points
    % fs: sampling frequency (in Hz)

    t = (0:num_samples-1) / fs; % Time vector

    ref_signals = []; % Initialize the reference signal matrix

    % Create sine and cosine signals for each harmonic
    for h = 1:harmonics
        ref_signals = [ref_signals; sin(2 * pi * h * frequency * t); cos(2 * pi * h * frequency * t)];
    end
end

% Calculate average accuracies
avg_cca_accuracy = mean(acc_cca) * 100;
avg_fbcca_accuracy = mean(acc_fbcca) * 100;

% Display the accuracies for each model
disp(['CCA Accuracy: ', num2str(avg_cca_accuracy), '%']);
disp(['FBCCA Accuracy: ', num2str(avg_fbcca_accuracy), '%']);

% Combine the accuracies into one matrix for plotting
accuracies = [avg_cca_accuracy, avg_fbcca_accuracy];

% Plot the accuracies on the same bar graph
figure;
bar(accuracies);
title('Classification Accuracies of CCA and FBCCA');
ylabel('Accuracy (%)');
xticklabels({'CCA', 'FBCCA'}); % Labels for x-axis
ylim([0, 100]); % Set the Y-axis range to 0-100%
set(gca, 'FontSize', 12); 

% data1_session2 = load('subject_1_fvep_led_training_2.mat');
% data2_session1 = load('subject_2_fvep_led_training_1.mat');
% data2_session2 = load('subject_2_fvep_led_training_2.mat');
