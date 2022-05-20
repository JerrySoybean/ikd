function result = exp6_pgplvm(spks,z_true,z_init,noise_variance)


[n_latents, d] = size(z_true);

z_init_aligned = align_xtrue(z_init,z_true);
%     r2_plds = corr(z_true(:),xpldsmat(:)).^2;
r2_plds = r2_score(z_true,z_init_aligned);

%     figure(1),clf
%     plot2d(xpldsmat,index_id,0)
figure(2),clf
subplot(211); plot(1:n_latents,z_true(:,1),'m-',1:n_latents,z_init_aligned(:,1),'k-','linewidth',2); drawnow; %legend('true x','PLDS x','P-GPLVM x','P-GPLVM old x');
subplot(212); plot(1:n_latents,z_true(:,2),'m-',1:n_latents,z_init_aligned(:,2),'k-','linewidth',2); drawnow;

%%
% Set up options
setopt.sigma2_init = noise_variance; % initial noise variance
setopt.lr = 0.95; % learning rate
setopt.latentTYPE = 2; % kernel for the latent, 1. AR1, 2. SE
setopt.ffTYPE = 2; % kernel for the tuning curve, 1. AR1, 2. SE
setopt.initTYPE = 1; % initialize latent: 1. use PLDS init; 2. use random init; 3. true xx
setopt.la_flag = 1; % 1. no la; 2. standard la; 3. decoupled la
setopt.rhoxx = 1; % rho for Kxx
setopt.lenxx = 5; % len for Kxx
setopt.rhoff = 1; % rho for Kff
setopt.lenff = 10; % len for Kff
setopt.b = 0; % rho for Kxx
setopt.r = 1; % rho for Kxx
setopt.nsevar = 100; % rho for Kxx
setopt.hypid = [1,5]; % 1. rho for Kxx; 2. len for Kxx; 3. rho for Kff; 4. len for Kff; 5. b for ff; 6. sigma2 (annealing it instead of optimizing it)
% setopt.xpldsmat = xllemat; % for plotting purpose
% setopt.xplds = xlle; % for initialization purpose
setopt.xpldsmat = z_init_aligned; % for plotting purpose
setopt.xplds = z_init; % for initialization purpose
setopt.niter = 50; % number of iterations
setopt.opthyp_flag = 0;

% == 3. Plot latent variables and tuning curves ====
% Initialize the log of spike rates with the square root of spike counts.
ffmat = sqrt(spks);
% ffmat = log(exp(yplds)+1);
xx = z_init;

%
latentTYPE = setopt.latentTYPE; % kernel for the latent, 1. AR1, 2. SE
ffTYPE = setopt.ffTYPE; % kernel for the tuning curve, 1. AR1, 2. SE
z_init_aligned = setopt.xpldsmat;
z_init = setopt.xplds;

% generate grid values as inducing points
tgrid = [1:n_latents]';
switch d
    case 1
        xgrid = gen_grid([min(xx(:,1)) max(xx(:,1))],25,d); % x grid (for plotting purposes)
    case 2
        xgrid = gen_grid([min(xx(:,1)) max(xx(:,1)); min(xx(:,2)) max(xx(:,2))],10,d); % x grid (for plotting purposes)
    case 3
        xgrid = gen_grid([min(xx(:,1)) max(xx(:,1)); min(xx(:,2)) max(xx(:,2)); min(xx(:,3)) max(xx(:,3))],5,d); % x grid (for plotting purposes)
end

% set hypers
hypers = [setopt.rhoxx; setopt.lenxx; setopt.rhoff; setopt.lenff]; % rho for Kxx; len for Kxx; rho for Kff; len for Kff

% set initial noise variance for simulated annealing
lr = setopt.lr; % learning rate
sigma2_init = setopt.sigma2_init;
propnoise_init = 0.001;
sigma2 = sigma2_init;
propnoise = propnoise_init;
b = setopt.b;
r = setopt.r;
nsevar = setopt.nsevar;

% set initial prior kernel
% K = Bfun(eye(n_latents),0)*Bfun(eye(n_latents),0)';
% Bfun maps the white noise space to xx space
[Bfun, BTfun, nu, sdiag] = prior_kernel(hypers(1),hypers(2),n_latents,latentTYPE,tgrid);
rhoxx = hypers(1); % marginal variance of the covariance function the latent xx
lenxx = hypers(2); % length scale of the covariance function for the latent xx
rhoff = hypers(3); % marginal variance of the covariance function for the tuning curve ff
lenff = hypers(4:end); % length scale of the covariance function for the tuning curve ff

% initialize latent
initTYPE = setopt.initTYPE;
switch initTYPE
    case 1  % use LLE or PPCA or PLDS init
        uu0 = Bfun(z_init,1);
        % uu0 = Bfun(xlle,1);
        % uu0 = Bfun(xppca,1);
    case 2   % use random init
        uu0 = randn(nu,d);%*0.01;
    case 3   % true xx
        uu0 = Bfun(z_init,1)+randn(nu,d);
end
uu = uu0;  % initialize sample
xxsamp = Bfun(uu,0);
xxsampmat = align_xtrue(xxsamp,xx);
xxsampmat_old = xxsampmat;
xxsamp_old = xxsamp;

% Now do inference
infTYPE = 1; % 1 for MAP; 2 for MH sampling; 3 for hmc
ppTYPE = 1; % 1 optimization for ff; 2. sampling for ff
la_flag = setopt.la_flag; % 1. no la; 2. standard la; 3. decoupled la
opthyp_flag = setopt.opthyp_flag; % flag for optimizing the hyperparameters

% set options for minfunc
options = [];
options.Method='lbfgs';
options.TolFun=1e-4;
options.MaxIter = 1e1;
options.maxFunEvals = 1e1;
options.Display = 'off';

options1 = [];
options1.Method='lbfgs';
options1.TolFun=1e-4;
options1.MaxIter = 1e1;
options1.maxFunEvals = 1e1;
options1.Display = 'off';

niter = setopt.niter;
clf
for iter = 1:niter
    
    if sigma2>1e-3
        sigma2 = sigma2*lr;  % decrease the noise variance with a learning rate
    end
    
    if ffTYPE==2
        lenff = median(vec(range(xxsamp)));
    end
    
    if ffTYPE==4
        lenff = vec(range(xxsamp));
    end
    
    %% 1. Find optimal ff
    [Bfun, BTfun, nu] = prior_kernel(rhoxx,lenxx,n_latents,latentTYPE,tgrid);
    covfun = covariance_fun(rhoff,lenff,ffTYPE); % get the covariance function
    cxx = covfun(xxsamp,xxsamp);
    incxx = pdinv(cxx+cxx(1,1)*sigma2*eye(size(cxx)));
    
    lmlifun_poiss = @(ff) StateSpaceModelsofSpikeTrains_noip(ff,spks,incxx);
    ff0 = vec(ffmat);
    floss_ff = @(ff) lmlifun_poiss(ff); % negative marginal likelihood
    %DerivCheck(floss_ff,ff0)
    [ffnew, fval] = minFunc(floss_ff,ff0,options);
    [L,dL,ffnew] = lmlifun_poiss(ffnew);
    ffmat = ffnew;
    
%     figure(1)
%     [~,yi] = max(sum(spks,1));
%     subplot(313),plot([spks(:,yi),exp(ffnew(:,yi))]),title(sigma2),legend('true ff','P-GPLVM ff'),drawnow
%     gg = size(spks,1);
%     tmp = spks(1:gg,:)./repmat(max(spks(1:gg,:))+1e-6,gg,1);
%     subplot(311), imagesc(tmp')
%     ff1 = exp(ffnew);
%     tmp = ff1(1:gg,:)./repmat(max(ff1(1:gg,:))+1e-6,gg,1);
%     subplot(312), imagesc(tmp')
%     drawnow
    %     figure(1),clf
    %     plot2d(z_true,index_id,0)
    
    %% 2. Find optimal latent xx, actually search in u space, xx=K^{1/2}*u
    uu = Bfun(xxsamp,1);
    switch ffTYPE
        case 1 % AR1 without grad
            % lmlifun = @(u) logmargli_gplvm_ar(u,Bfun,ffmat,covfun,sigma2,d); % only works for 1d
            lmlifun = @(u) logmargli_gplvm_se(u,Bfun,ffmat,covfun,sigma2,d);
        case {2,4} % SE with grad
            switch la_flag
                case 1
                    % no la, poisson
                    lmlifun = @(u) logmargli_gplvm_se_block(u,Bfun,ffmat,covfun,sigma2,d,BTfun,b);
                case 2
                    % standard la
                    lmlifun = @(u) logmargli_gplvm_se_la(u,Bfun,ffmat,covfun,sigma2,d,BTfun);
                case 3
                    % decouple la
                    lmlifun = @(u) logmargli_gplvm_se_sor_la_decouple(u,spks,Bfun,ffmat,covfun,sigma2,d,BTfun,xgrid,cuu,cuuinv,cufx_old,invcc_old);
            end
    end
    
    % set up MAP inference
    floss = @(u) lmlifun(vec(u));
    %DerivCheck(floss,vec(uu))
    uunew = minFunc(floss,vec(uu),options1);
    %lmlifun = @(u) logmargli_gplvm_se_la(u,Bfun,ffmat,covfun,sigma2,d,BTfun);
    [~,~,lm] = lmlifun(vec(uunew));
    uu = reshape(uunew,[],d);
    xxsamp = Bfun(uu,0);
    
    % plot latent xx
    xxsampmat = align_xtrue(xxsamp,z_true);
%         r2_pgplvm = corr(z_true(:),xxsampmat(:)).^2;
    r2_pgplvm = r2_score(z_true,xxsampmat);
    
    figure(2),clf
    subplot(411); plot(1:n_latents,z_true(:,1),'b-',1:n_latents,z_init_aligned(:,1),'m-',1:n_latents,xxsampmat(:,1),'k-',1:n_latents,xxsampmat_old(:,1),'k:','linewidth',2); drawnow;
    legend('true x','PLDS x','P-GPLVM x','P-GPLVM old x');
    subplot(412); plot(1:n_latents,z_true(:,2),'b-',1:n_latents,z_init_aligned(:,2),'m-',1:n_latents,xxsampmat(:,2),'k-',1:n_latents,xxsampmat_old(:,2),'k:','linewidth',2); drawnow;
    subplot(212),cla
%     plot2d(xxsamp,index_id,0)
    title(['    plds r2: ' num2str(r2_plds) '   pgplvm r2:' num2str(r2_pgplvm)])
    drawnow
    
    xxsampmat_old = xxsampmat;
    xxsamp_old = xxsamp;
    display(['iter:' num2str(iter) ', lm:' num2str(lm) ', rhoxx:' num2str(rhoxx) ', lenxx:' num2str(lenxx) ', rhoff:' num2str(rhoff) ', lenff:' num2str(vec(lenff)') ', nsevar:' num2str(nsevar) ', b:' num2str(b) ', sigma2:' num2str(sigma2)])
    
end
        
result.xxsamp = xxsamp;
result.xxsampmat = xxsampmat;
result.ffmat = ffmat;
result.rhoxx = rhoxx;
result.lenxx = lenxx;
result.rhoff = rhoff;
result.lenff = lenff;
result.b = b;
end

