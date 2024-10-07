%% Data Preprocessing

% Load session data
data1_session1 = load('subject_1_fvep_led_training_1.mat');

% Load class information from 'classInfo_4_5.m' file
class_labels = [
    1 0 0 0;
    0 1 0 0;
    0 0 1 0;
    0 0 0 1;
    1 0 0 0;
    0 1 0 0;
    0 0 1 0;
    0 0 0 1;
    1 0 0 0;
    0 1 0 0;
    0 0 1 0;
    0 0 0 1;
    1 0 0 0;
    0 1 0 0;
    0 0 1 0;
    0 0 0 1;
    1 0 0 0;
    0 1 0 0;
    0 0 1 0;
    0 0 0 1
];

% Convert one-hot encoded labels to frequencies
ground_truth_frequencies = [9, 10, 12, 15];  % Define this to match the order of the one-hot encoding
ground_truth_frequencies = ground_truth_frequencies(class_labels * [1; 2; 3; 4]);

% Create a new variable for comparing with LDA
lda_ground_truth = zeros(length(ground_truth_frequencies), 1);  % Initialize LDA ground truth
for i = 1:length(ground_truth_frequencies)
    if ismember(ground_truth_frequencies(i), [9, 10])
        lda_ground_truth(i) = 0;  % Low-frequency group (9/10 Hz)
    elseif ismember(ground_truth_frequencies(i), [12, 15])
        lda_ground_truth(i) = 3;  % High-frequency group (12/15 Hz)
    end
end

% Extract parts 
sample_time = data1_session1.y(1, 1001:end); % Time samples from the 1001st sample onward
EEG_data = data1_session1.y(2:9, 1001:end); % EEG data from CH2-9, starting from the 1001st sample
trigger_info = data1_session1.y(10, 1001:end); % Trigger information from 1001st sample
lda_output = data1_session1.y(11, 1001:end); % LDA classification output from 1001st sample

% Sampling rate
fs = 256; % Hz

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

% Find the transitions in trigger_info from 0 to 1
onsets = find(diff(trigger_info) == 1) + 1;  % +1 to get the first sample of the onset

% Initialize epochs container
epochs = {}; 
window_size = fs * 3;  % Example window of 3 seconds

% Extract the epochs around the stimulation onsets
for i = 1:length(onsets)
    if (onsets(i) + window_size - 1) <= length(sample_time)
        epoch = EEG_data_filtered(:, onsets(i):onsets(i) + window_size - 1);
        epochs{i} = epoch;  % Store the epoch
    end
end

% Ensure that lda_output matches the number of epochs
num_epochs = length(epochs);
assert(num_epochs == length(lda_ground_truth))

% Define frequencies of interest and harmonics
frequencies_of_interest = [9, 10, 12, 15];  % Add these before the parfor
frequencies_of_interest_reversed = flip(frequencies_of_interest);
harmonics = 3;  % Number of harmonics to consider

%% Plot the FFT of each epoch

% Parameters for FFT
nfft = window_size;  % FFT points should match the length of the window
frequencies = (0:nfft-1) * (fs / nfft);  % Frequency axis (in Hz)
max_plot_freq = 60;  % Maximum frequency to plot (e.g., up to 60 Hz)

% Plot the FFT for each channel in each epoch
for i = 1:length(epochs)
    epoch = epochs{i};
    
    figure;
    sgtitle(['FFT for Epoch ', num2str(i)]);  % Title for the plot
    
    for ch = 1:size(epoch, 1)  % Loop over all EEG channels
        % Compute FFT
        epoch_fft = fft(epoch(ch, :), nfft);
        P2 = abs(epoch_fft / nfft);  % Two-sided spectrum
        P1 = P2(1:nfft/2+1);  % One-sided spectrum
        P1(2:end-1) = 2 * P1(2:end-1);  % Double the energy for the positive half
        
        % Plot
        subplot(4, 2, ch);  % Assuming 8 channels, create a subplot for each channel
        plot(frequencies(1:nfft/2+1), P1);
        title(['Channel ', num2str(ch)]);
        xlabel('Frequency (Hz)');
        ylabel('Amplitude');
        xlim([0 max_plot_freq]);  % Limit the x-axis to the maximum plot frequency
        grid on;
    end
end


%% LDA Alignment using Trigger Info

% Extract LDA output for each epoch based on trigger transitions
predicted_labels_lda = zeros(num_epochs, 1);  % Initialize the predicted LDA labels

for i = 1:num_epochs
    epoch_start = onsets(i);  % Get the start of the epoch based on trigger_info
    epoch_end = min(epoch_start + window_size - 1, length(lda_output));  % Ensure we don't exceed the LDA output length

    % Extract the LDA output for the duration of the epoch
    epoch_lda_output = lda_output(epoch_start:epoch_end);
    
    % Use non-zero values of LDA output to find relevant predictions
    non_zero_lda_output = epoch_lda_output(epoch_lda_output ~= 0);
    
    % If there are non-zero values, take the most frequent (mode) or the last value
    if ~isempty(non_zero_lda_output)
        predicted_labels_lda(i) = mode(non_zero_lda_output);  % Use mode for stable predictions
    else
        predicted_labels_lda(i) = 0;  % Fallback to 0 if no non-zero LDA values are found
    end
end

%% CCA, FBCCA, and LDA Analysis

acc_cca = zeros(num_epochs, 1);
acc_fbcca = zeros(num_epochs, 1);
acc_lda = zeros(num_epochs, 1);  
predicted_labels_cca = zeros(num_epochs, 1); 
predicted_labels_fbcca = zeros(num_epochs, 1); 

% Check if a pool already exists
if isempty(gcp('nocreate'))
    parpool;
end

parfor i = 1:num_epochs
    epoch = epochs{i};

    % Ensure we do not have low variance
    assert(var(epoch(:)) >= 1e-10);

    % Apply PCA
    [coeff, score, ~] = pca(epoch'); 
    reduced_epoch = score(:, 1:min(size(score, 2), 4)); 

    % --- CCA Analysis ---
    max_corr = zeros(1, length(frequencies_of_interest));
    for f_idx = 1:length(frequencies_of_interest)
        ref_signals = create_reference_signals(frequencies_of_interest_reversed(f_idx), harmonics, size(reduced_epoch, 1), fs);
        [~, ~, r] = canoncorr(reduced_epoch, ref_signals');
        max_corr(f_idx) = max(r); 
    end
    [~, predicted_freq_idx_cca] = max(max_corr); 
    predicted_labels_cca(i) = frequencies_of_interest(predicted_freq_idx_cca);

    % Check prediction
    acc_cca(i) = (predicted_labels_cca(i) == ground_truth_frequencies(i));

    % --- FBCCA Analysis ---
    fb_corr = zeros(1, length(frequencies_of_interest));
    num_bands = 7;
    for band_idx = 1:num_bands
        f_lower = max(0, frequencies_of_interest - 2 * band_idx);  % Clamp to 0
        f_upper = min(fs/2, frequencies_of_interest + 2 * band_idx);  % Clamp to Nyquist
    
        for f_idx = 1:length(frequencies_of_interest)
            % Ensure valid frequency range for bandpass filter
            if f_lower(f_idx) >= f_upper(f_idx)
                warning('Skipping band %d due to invalid frequency range.', band_idx);
                continue;
            end
    
            % Normalize frequencies to be between 0 and 1
            norm_f_lower = f_lower(f_idx) / (fs/2);
            norm_f_upper = f_upper(f_idx) / (fs/2);
    
            % Ensure that the normalized frequencies are within the valid range (0,1)
            if norm_f_lower <= 0 || norm_f_upper >= 1 || norm_f_lower >= norm_f_upper
                warning('Skipping due to invalid normalized frequency range.');
                continue;
            end
    
            % Apply Butterworth filter with validated frequency range
            [b_fb, a_fb] = butter(1, [norm_f_lower norm_f_upper], 'bandpass');
            epoch_fb = filtfilt(b_fb, a_fb, epoch);
    
            % CCA analysis on filtered epoch
            ref_signals = create_reference_signals(frequencies_of_interest_reversed(f_idx), harmonics, size(epoch_fb, 2), fs);
            [~, ~, r_fb] = canoncorr(epoch_fb', ref_signals');
            fb_corr(f_idx) = max(r_fb) + fb_corr(f_idx);
        end
    end
    
    [~, predicted_freq_idx_fbcca] = max(fb_corr); 
    predicted_labels_fbcca(i) = frequencies_of_interest(predicted_freq_idx_fbcca);

    % Check prediction
    acc_fbcca(i) = (predicted_labels_fbcca(i) == ground_truth_frequencies(i));

    % --- LDA Analysis ---
    acc_lda(i) = (predicted_labels_lda(i) == lda_ground_truth(i));

    % Progress
    if mod(i, 10) == 0
        fprintf('Epoch %d of %d processed.\n', i, num_epochs);
    end
end

delete(gcp); % Shutdown parallel pool

% --- Helper Function to Create Reference Signals ---
function ref_signals = create_reference_signals(frequency, harmonics, num_samples, fs)
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
avg_lda_accuracy = mean(acc_lda) * 100;

% Display the accuracies for each model
disp(['CCA Accuracy: ', num2str(avg_cca_accuracy), '%']);
disp(['FBCCA Accuracy: ', num2str(avg_fbcca_accuracy), '%']);
disp(['LDA Accuracy: ', num2str(avg_lda_accuracy), '%']);

% Combine the accuracies into one matrix for plotting
accuracies = [avg_cca_accuracy, avg_fbcca_accuracy, avg_lda_accuracy];

% Plot the accuracies on the same bar graph
figure;
bar(accuracies);
title('Classification Accuracies of CCA, FBCCA, and LDA');
ylabel('Accuracy (%)');
xticklabels({'CCA', 'FBCCA', 'LDA'}); % Labels for x-axis
ylim([0, 100]); % Set the Y-axis range to 0-100%
set(gca, 'FontSize', 12); 