clear x;

% Symbols:
% 1-S;2-R;3-N;4-E;5-ITI;0-Gap
% rng(0);

% rng(1);

% trial_prototype(1,:)=[1,0,0,0,2,2,0,0,0,3,3,0,0,0,2,2,0,0,0,4,5,5]; %RNR
% trial_prototype(2,:)=[1,0,0,0,2,2,0,0,0,2,2,0,0,0,3,3,0,0,0,4,5,5]; %RRN
% trial_prototype(3,:)=[1,0,0,0,3,3,0,0,0,2,2,0,0,0,2,2,0,0,0,4,5,5]; %NRR

trial_prototype(1,:)=[0,1,2,1]; %RNR
trial_prototype(2,:)=[0,1,1,2]; %RRN
trial_prototype(3,:)=[0,2,1,1]; %NRR

n_step=numel(trial_prototype(1,:));

n_trials=5000;

p = [1, 1, 1]/3;

for i=1:n_trials
    r = rand();           % Uniform(0,1)
    idx = find(r <= cumsum(p),1);
    if (i==1)
        x=trial_prototype(idx,:);
    else
        x((end+1):(end+n_step))=trial_prototype(idx,:);
    end
end

x_color=[0.4588, 0.7059, 0.5725;0.8549, 0.5647, 0.6824;0.4941, 0.6941, 0.7922];

% x_color=[0.459,0.706,0.57;1,0.4,0;1,0.7,0;1,1,0]