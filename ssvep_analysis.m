% Load session data
data1_session1 = load('subject_1_fvep_led_training_1.mat');

% Extract parts 
sample_time = data1_session1.y(1, 1001:end); % Time samples from the 1001st sample onward
EEG_data = data1_session1.y(2:9, 1001:end); % EEG data from CH2-9, starting from the 1001st sample
trigger_info = data1_session1.y(10, 1001:end); % Trigger information from 1001st sample
lda_output = data1_session1.y(11, 1001:end); % LDA classification output from 1001st sample

% Sampling rate
fs = 256; % Sampling rate in Hz

% Bandpass filter settings (0.5-30 Hz)
[b_bandpass, a_bandpass] = butter(4, [0.5 30]/(fs/2), 'bandpass'); % 4th-order Butterworth bandpass filter

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
window_size = fs * 3; % For example, a 3-second window (adjust as needed)
onsets = find(trigger_info == 1); % Find where the stimulus is ON (trigger == 1)

% Extract epochs based on the stimulus onsets
for i = 1:length(onsets)
    if (onsets(i) + window_size - 1) <= length(sample_time)
        epochs{i} = EEG_data_filtered(:, onsets(i):onsets(i) + window_size - 1);
    end
end

% Frequencies of interest for SSVEP stimuli
frequencies_of_interest = [9, 10, 12, 15]; % Define target frequencies
harmonics = 2; % Number of harmonics to consider for CCA
num_channels = size(EEG_data_filtered, 1);

% CCA and FBCCA analysis
acc_cca = [];
acc_fbcca = [];

for i = 1:length(epochs)
    epoch = epochs{i}; % Get the i-th trial/epoch
    
    % --- CCA ---
    max_corr = zeros(1, length(frequencies_of_interest));
    for f_idx = 1:length(frequencies_of_interest)
        ref_signals = create_reference_signals(frequencies_of_interest(f_idx), harmonics, size(epoch, 2), fs);
        [A, B, r] = canoncorr(epoch', ref_signals'); % CCA between EEG epoch and reference signals
        max_corr(f_idx) = max(r); % Take the maximum correlation
    end
    [~, predicted_freq_idx_cca] = max(max_corr); % Predicted frequency (CCA)
    acc_cca = [acc_cca; predicted_freq_idx_cca]; % Store the result
    
    % --- FBCCA (Filter Bank CCA) ---
    num_bands = 5; % Number of frequency bands for FBCCA
    fb_corr = zeros(1, length(frequencies_of_interest));
    for band_idx = 1:num_bands
        % Create bandpass filters for each band (reduce the filter order to 1)
        f_lower = frequencies_of_interest - 2 * band_idx; % Lower edge
        f_upper = frequencies_of_interest + 2 * band_idx; % Upper edge
        [b_fb, a_fb] = butter(1, [f_lower(f_idx) f_upper(f_idx)]/(fs/2), 'bandpass'); % Filter order reduced to 1
        
        % Zero-pad the epoch if it's too short for the filter
        if size(epoch, 2) < 3 * length(b_fb)
            epoch_padded = [epoch, zeros(size(epoch, 1), 3 * length(b_fb) - size(epoch, 2))]; % Zero-pad the epoch
        else
            epoch_padded = epoch;
        end
        
        epoch_fb = filtfilt(b_fb, a_fb, epoch_padded); % Apply filter to the padded epoch
        
        % CCA with filter bank
        for f_idx = 1:length(frequencies_of_interest)
            ref_signals = create_reference_signals(frequencies_of_interest(f_idx), harmonics, size(epoch_fb, 2), fs);
            [~, ~, r_fb] = canoncorr(epoch_fb', ref_signals'); % CCA between filtered EEG and reference signals
            fb_corr(f_idx) = max(r_fb) + fb_corr(f_idx); % Accumulate correlation over bands
        end
    end
    [~, predicted_freq_idx_fbcca] = max(fb_corr); % Predicted frequency (FBCCA)
    acc_fbcca = [acc_fbcca; predicted_freq_idx_fbcca]; % Store the result
end

% Plot Classification Accuracy for CCA
figure;
bar(mean(acc_cca)); % Plot accuracy as a bar graph
title('CCA Classification Accuracy');
ylabel('Accuracy (%)');
xlabel('Cross-Validated CCA');
ylim([0, 100]); % Set the Y-axis range to 0-100%
set(gca, 'FontSize', 12); % Increase font size for readability

% Plot Classification Accuracy for FBCCA
figure;
bar(mean(acc_fbcca)); % Plot accuracy as a bar graph
title('FBCCA Classification Accuracy');
ylabel('Accuracy (%)');
xlabel('Cross-Validated FBCCA');
ylim([0, 100]); % Set the Y-axis range to 0-100%
set(gca, 'FontSize', 12); % Increase font size for readability

% Add additional helper function for creating reference signals
function ref_signals = create_reference_signals(freq, harmonics, num_samples, fs)
    % Generate reference sinusoidal signals for CCA
    ref_signals = [];
    t = (0:num_samples-1)/fs;
    for h = 1:harmonics
        ref_signals = [ref_signals; sin(2*pi*h*freq*t); cos(2*pi*h*freq*t)];
    end
end

% data1_session2 = load('subject_1_fvep_led_training_2.mat');
% data2_session1 = load('subject_2_fvep_led_training_1.mat');
% data2_session2 = load('subject_2_fvep_led_training_2.mat');