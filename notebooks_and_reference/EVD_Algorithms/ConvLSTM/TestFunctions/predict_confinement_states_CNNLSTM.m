function objTB = predict_confinement_states_CNNLSTM(shots)

addpath('../data_access');
data_dir = '../data/Detected/';

name_device = 'TCV';
for i=1:numel(shots)
    shot = shots(i);
        % Prepare cvs_type
    files_signals_search = fullfile('data','Detected',sprintf('%s_%d_signals.csv',name_device,shot));
    disp(files_signals_search);
    if ~exist(files_signals_search, 'file')
            % Get inputs
        objTB.sigs = get_sig_data_TCV(shot,0);
            % Store inputs in a CSV table
        csv_handling('w','signals',data_dir,shot,name_device,objTB,'verbose',1);
    end
    
    %% Predict Plasma States with CNNLSTM
    cmd = sprintf('python algorithms/ConvLSTM/evaluate_model_from_detected_signals.py baseline_16042021_exp9 400 %d %s avg True',...
        shot,name_device);
    system(cmd);
end
