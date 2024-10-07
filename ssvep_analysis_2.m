%% Data Preprocessing for Both Subjects and Sessions

subjects = {'subject_1', 'subject_2'};
sessions = {'training_1', 'training_2'};
fs = 256; % Sampling rate in Hz

% Define frequencies of interest and harmonics
frequencies_of_interest = [9, 10, 12, 15];  % Frequencies corresponding to the task
frequencies_of_interest_reversed = flip(frequencies_of_interest); % Reversed for FBCCA analysis
harmonics = 3;  % Number of harmonics to consider

for subj = 1:length(subjects)
    for sess = 1:length(sessions)
        % Load session data
        data = load([subjects{subj} '_fvep_led_' sessions{sess} '.mat']);
        
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

        % Create LDA ground truth
        lda_ground_truth = zeros(length(ground_truth_frequencies), 1);
        for i = 1:length(ground_truth_frequencies)
            if ismember(ground_truth_frequencies(i), [9, 10])
                lda_ground_truth(i) = 0;  % Low-frequency group
            elseif ismember(ground_truth_frequencies(i), [12, 15])
                lda_ground_truth(i) = 3;  % High-frequency group
            end
        end

        % Extract data
        sample_time = data.y(1, 1001:end);  % Time samples from 1001st onward
        EEG_data = data.y(2:9, 1001:end);  % EEG data from CH2-9, starting from 1001st sample
        trigger_info = data.y(10, 1001:end);  % Trigger information
        lda_output = data.y(11, 1001:end);  % LDA output

        % Bandpass and Notch filter design
        [b_bandpass, a_bandpass] = butter(4, [5 30]/(fs/2), 'bandpass'); % 4th-order Butterworth
        [b_notch, a_notch] = butter(2, [49 51]/(fs/2), 'stop'); % 2nd-order Notch filter
        
        % Filter EEG data
        EEG_data_filtered = zeros(size(EEG_data));
        for ch = 1:8
            EEG_data_bandpassed = filtfilt(b_bandpass, a_bandpass, EEG_data(ch, :));
            EEG_data_filtered(ch, :) = filtfilt(b_notch, a_notch, EEG_data_bandpassed);
        end
        
        % Find transitions in trigger_info
        onsets = find(diff(trigger_info) == 1) + 1;

        % Initialize epochs
        epochs = {}; 
        window_size = fs * 3;  % 3-second windows
        
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

        num_epochs = length(epochs);
        assert(num_epochs == length(lda_ground_truth))  % Ensure match between epochs and labels

        % Perform CCA, FBCCA, and LDA Analysis for each epoch
        acc_cca = zeros(num_epochs, 1);
        acc_fbcca = zeros(num_epochs, 1);
        acc_lda = zeros(num_epochs, 1);  
        predicted_labels_cca = zeros(num_epochs, 1); 
        predicted_labels_fbcca = zeros(num_epochs, 1); 

        if isempty(gcp('nocreate'))
            parpool; % Open parallel pool if not already opened
        end

        parfor i = 1:num_epochs
            epoch = epochs{i};

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

            acc_cca(i) = (predicted_labels_cca(i) == ground_truth_frequencies(i));

            % --- FBCCA Analysis ---
            num_bands = 7;
            fb_corr = zeros(1, length(frequencies_of_interest));
            for band_idx = 1:num_bands
                f_lower = max(0, frequencies_of_interest - 2 * band_idx);  
                f_upper = min(fs/2, frequencies_of_interest + 2 * band_idx);  

                for f_idx = 1:length(frequencies_of_interest)
                    norm_f_lower = f_lower(f_idx) / (fs/2);
                    norm_f_upper = f_upper(f_idx) / (fs/2);

                    if norm_f_lower <= 0 || norm_f_upper >= 1 || norm_f_lower >= norm_f_upper
                        continue;
                    end

                    [b_fb, a_fb] = butter(1, [norm_f_lower norm_f_upper], 'bandpass');
                    epoch_fb = filtfilt(b_fb, a_fb, epoch);

                    ref_signals = create_reference_signals(frequencies_of_interest_reversed(f_idx), harmonics, size(epoch_fb, 2), fs);
                    [~, ~, r_fb] = canoncorr(epoch_fb', ref_signals');
                    fb_corr(f_idx) = max(r_fb) + fb_corr(f_idx);
                end
            end
            [~, predicted_freq_idx_fbcca] = max(fb_corr); 
            predicted_labels_fbcca(i) = frequencies_of_interest(predicted_freq_idx_fbcca);
            acc_fbcca(i) = (predicted_labels_fbcca(i) == ground_truth_frequencies(i));

            acc_lda(i) = (predicted_labels_lda(i) == lda_ground_truth(i));
        end
        
        if subj == 2 && sess == 2
            delete(gcp);  % Shutdown parallel pool
        end

        % Calculate and display average accuracies
        avg_cca_accuracy = mean(acc_cca) * 100;
        avg_fbcca_accuracy = mean(acc_fbcca) * 100;
        avg_lda_accuracy = mean(acc_lda) * 100;
        disp(['CCA Accuracy for ' subjects{subj} ', ' sessions{sess} ': ', num2str(avg_cca_accuracy), '%']);
        disp(['FBCCA Accuracy for ' subjects{subj} ', ' sessions{sess} ': ', num2str(avg_fbcca_accuracy), '%']);
        disp(['LDA Accuracy for ' subjects{subj} ', ' sessions{sess} ': ', num2str(avg_lda_accuracy), '%']);

        % Calculate and display average accuracies
        avg_cca_accuracy = mean(acc_cca) * 100;
        avg_fbcca_accuracy = mean(acc_fbcca) * 100;
        avg_lda_accuracy = mean(acc_lda) * 100;
        disp(['CCA Accuracy for ' subjects{subj} ', ' sessions{sess} ': ', num2str(avg_cca_accuracy), '%']);
        disp(['FBCCA Accuracy for ' subjects{subj} ', ' sessions{sess} ': ', num2str(avg_fbcca_accuracy), '%']);
        disp(['LDA Accuracy for ' subjects{subj} ', ' sessions{sess} ': ', num2str(avg_lda_accuracy), '%']);
        
        % Replace underscores with spaces and capitalize first letters
        subject_title = strrep(subjects{subj}, '_', ' '); % Replace underscore with space
        subject_title = ['Subject ' subject_title(end)]; % Modify to "Subject 1" format
        session_title = strrep(sessions{sess}, '_', ' '); % Replace underscore with space
        session_title = ['Training ' session_title(end)]; % Modify to "Training 1" format

        % Plot the accuracies
        accuracies = [avg_cca_accuracy, avg_fbcca_accuracy, avg_lda_accuracy];
        figure;
        bar(accuracies);
        title(['Accuracies for ' subject_title ', ' session_title]); % Set the title using the modified strings
        ylabel('Accuracy (%)');
        xticklabels({'CCA', 'FBCCA', 'LDA'}); 
        ylim([0, 100]);
        set(gca, 'FontSize', 12);
    end
end


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