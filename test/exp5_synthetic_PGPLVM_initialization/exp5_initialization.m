function exp5_initialization(trial)
addpath(genpath(pwd)); warning off
load exp5_py2mat.mat exp5_py2mat

z_true = exp5_py2mat{trial}.z_true;
spikes = exp5_py2mat{trial}.spikes;
firing_rates = exp5_py2mat{trial}.firing_rates;
firing_rates_est = exp5_py2mat{trial}.firing_rates_est;
d = size(z_true, 2);

%% run_plds
z_plds = run_plds(spikes, d, 'NucNormMin')';
%     function paramsCXd = computeCXd(z, firing_rates)
%         n_latents = size(firing_rates, 1);
%         X_aug = [ones(n_latents, 1), z];
%         Cd = X_aug \ firing_rates;
%         paramsCXd.Xpca = z';
%         paramsCXd.C = Cd(2:end,:)';
%         paramsCXd.d = Cd(1,:)';
%     end
%     z_plds_nnorm = run_plds(spikes, d, 'NucNormMin')';
%     z_plds_epca = run_plds(spikes, d, 'ExpFamPCA')';
%     z_plds_block = run_plds(spikes, d, 'params', computeCXd(z_block, sqrt(spikes)))';
%     z_plds_block_denoised = run_plds(spikes, d, 'params', computeCXd(z_block_denoised, sqrt(firing_rates_est)))';
exp5_py2mat{trial}.z_plds = z_plds;

%% PGPLVM
setopt.sigma2_init = 0.02; % initial noise variance
setopt.lr = 0.95; % learning rate
setopt.latentTYPE = 1; % kernel for the latent, 1. AR1, 2. SE
setopt.ffTYPE = 2; % kernel for the tuning curve, 1. AR1, 2. SE
setopt.la_flag = 1; % 1. no la; 2. standard la; 3. decoupled la
setopt.rhoxx = 10; % rho for Kxx
setopt.lenxx = 50; % len for Kxx
setopt.rhoff = 1; % rho for Kff
setopt.lenff = 1; % len for Kff
setopt.hypid = [1,2,3,4]; % 1. rho for Kxx; 2. len for Kxx; 3. rho for Kff; 4. len for Kff; 5. sigma2 (annealing it instead of optimizing it)
setopt.niter = 50; % number of iterations
setopt.initTYPE = 1; % initialize latent: 1. use PLDS/given init; 2. use random init; 3. true xx

names = {'pca'; 'kernel_pca'; 'le'; 'epca'; 'plds'; 'ikd'; 'eikd'; 'ikd_b'; 'eikd_b'};
result = struct();
for init = 1:numel(names)
    z_init = exp5_py2mat{trial}.(['z_', names{init}]);
    result.(names{init}).init = z_init;

    setopt.xplds = z_init;
    setopt.xpldsmat = align_xtrue(z_init, z_true);
    result_la = pgplvm_la(spikes,z_true,firing_rates,setopt);
    result.(names{init}).pgplvm = result_la.xxsamp;
end

%% Save result for Python
save(sprintf("outputs/exp5_mat2py_%d.mat", trial),"result");
end