function exp6_empirical_spikes(trial, noise_variance)
addpath(genpath(pwd)); warning off
load exp6_py2mat.mat exp6_py2mat

magnitude = @(z) mean(sqrt(sum(z.^2, 2)));

z_true = exp6_py2mat{trial}.z_true;
spikes = exp6_py2mat{trial}.spikes;
% firing_rates_est = exp6_py2mat{trial}.firing_rates_est;
[n_latents,d] = size(z_true);
n_neurons = size(spikes, 2);

%% train-infer-test split, for PLL
n_latents_train = round(n_latents*0.9);
n_neurons_infer = round(n_neurons*0.7);

z_true_train = z_true(1:n_latents_train,:);
spikes_train = spikes(1:n_latents_train,:);
spikes_infer = spikes(n_latents_train+1:end,1:n_neurons_infer);
spikes_test = spikes(n_latents_train+1:end,n_neurons_infer+1:end);

%% for R^2, PLDS initialization
z_plds = run_plds(spikes,d)';
exp6_py2mat{trial}.z_plds = z_plds;

%% PLL of PLDS
options = [];
options.Method='scg';
options.TolFun=10;
options.MaxIter = 1e2;
options.Display = 'off';

%% prepare for saving
names = {'pca'; 'kernel_pca'; 'le'; 'epca'; 'plds'; 'ikd'; 'eikd'};
result = struct();

for init = 1:numel(names)
    z_init = exp6_py2mat{trial}.(['z_', names{init}]);
    z_init = z_init / magnitude(z_init);
    result.(names{init}).init = z_init;
    z_train_init = z_init(1:n_latents_train,:);
    pgplvm_result = exp6_pgplvm(spikes,z_true,z_init,noise_variance);
    result.(names{init}).pgplvm = pgplvm_result.xxsamp;
    
    pgplvm_result = exp6_pgplvm(spikes_train,z_true_train,z_train_init,noise_variance);
    L_pred_infer = @(xpred) ll_y_fx(vec(xpred),pgplvm_result.xxsamp,pgplvm_result.ffmat(:,1:n_neurons_infer),spikes_infer,pgplvm_result.rhoxx,pgplvm_result.lenxx,pgplvm_result.rhoff,pgplvm_result.lenff,2,1);
    T = [ones(n_latents_train, 1) z_train_init] \ pgplvm_result.xxsamp;
    z_infer_test_init = [ones(n_latents - n_latents_train, 1) z_init(n_latents_train+1:end,:)] * T;
    z_infer_test = minFunc(L_pred_infer,vec(z_infer_test_init),options);
    result.(names{init}).pll = -ll_y_fx(vec(z_infer_test),pgplvm_result.xxsamp,pgplvm_result.ffmat(:,n_neurons_infer+1:end),spikes_test,pgplvm_result.rhoxx,pgplvm_result.lenxx,pgplvm_result.rhoff,pgplvm_result.lenff,2,0);
end
result.z_true = z_true;
result.trial = trial - 1;

dir_name = sprintf('outputs_%g', noise_variance);
mkdir(dir_name);
save(sprintf([dir_name, '/exp6_mat2py_%s_%s_%s.mat'],exp6_py2mat{trial}.mouse_name,...
        num2str(exp6_py2mat{trial}.day), num2str(exp6_py2mat{trial}.epoch)), "result");
end