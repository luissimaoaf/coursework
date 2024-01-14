clear;
clc;

% Importing the datasets
% These are obtained through the python package yfinance
data = readtable('sp500.csv');
% Obtaining the dates
dates = datetime(data.Date); 
dates(1) = [];

% Adjusted closing prices
price = data.AdjClose;
% Compute log returns
yt = 100 * (log(price(2:end)) - log(price(1:end-1)));

% Sample size
T = size(yt, 1); 


%% Preliminary analysis

% Index plot
figure('position', [355   320   800   400]);
plot(dates, price(2:end))
ylabel('S&P 500 index');

% Plot ACFs and stationarity statistics
figure('position', [355   320   800   400]);
plotcorrstat(dates, yt, 30, 1:30)
subplot(2, 2, 1);
ylabel('Returns');
figure('position', [355   320   800   400]);
plotcorrstat(dates, yt.^2, 30, 1:30)
subplot(2, 2, 1);
ylabel('Returns^2');


%% Estimate GARCH(1,1) model

% Defining the model
arima_model = arima(2, 0, 2);
arima_model.Variance = garch(1, 1);
% Model estimation, extracting residuals and variance
[estimated_model, EstParamCov, logL, info] = estimate(arima_model, yt);
[errors, GARCH_vol, logL] = infer(estimated_model, yt);
residuals = errors ./ sqrt(GARCH_vol);


%% Residual analysis and volatility plot

% Plot ACFs on residuals
figure('position', [355   320   800   400]);
plotcorrstat(dates, residuals, 30, 1:30)
subplot(2, 2, 1);
ylabel('\epsilon_t');

% Plot squared residuals' ACFs
figure('position', [355   320   800   400]);
plotcorrstat(dates, residuals.^2, 30, 1:30)
subplot(2, 2, 1);
ylabel('\epsilon_t^2');


%% Setting the priors for MCMC estimation

% Normal prior for mu
m0 = 0.05;
sigma2_m0 = 0.25;
% Normal prior for phi0
a0 = -0.01;
A0 = 0.2;
% Beta prior for phi1
rho1 = 20;
rho2 = 1.5;
% Inverse gamma prior for sigma2_v
nu0 = 1;
Sigmav0 = 0.01;
nu1 = nu0 + T - 1;
% Initial estimates using the GARCH(1,1) model
% and least square fit of log(ht0)
mu_draw = estimated_model.Constant;
h_draw = log(GARCH_vol);

least_squares = fitlm(lagmatrix(h_draw, 1), h_draw);
phi_draw = least_squares.Coefficients{:, 1}';

sigmav_draw = var(least_squares.Residuals.Raw, "omitnan");

% Variance of the normal distribution proposed for new ht values
sigma0_h = 0.1;

%% Initialization of the MCMC algorithm
burnin = 150000;
sample_size = 250000;
n_sims = burnin + sample_size;

% Output dataframes
mu_sample = zeros(n_sims, 1);
h_sample = zeros(T, n_sims);
phi_sample = zeros(n_sims, 2);
sigmav_sample = zeros(n_sims, 1);

% Number of accepted candidate draws in Metropolis-Hastings
AccDraws_h = 0;
AccDraws_phi = 0;

%% Markov Chain Monte Carlo

tic

for iter = 1:n_sims
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 1) Sample mu by Bayesian linear regression
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    vol_t = exp(h_draw);
    yt_star = yt ./ sqrt(vol_t);
    xt = 1 ./ sqrt(vol_t);

    sigma2_m1 = 1 / (1/sigma2_m0 + xt'*xt);
    m1 = sigma2_m1*(m0/sigma2_m0 + xt'*yt_star);
    mu_draw = normrnd(m1, sqrt(sigma2_m1));
    mu_sample(iter) = mu_draw;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 2) Draw h_t by independence Metropolis-Hastings
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    eta = phi_draw(1)/(1 - phi_draw(2));
    % One-step backwards prediction
    h0_hat = phi_draw(2)^2*(h_draw(2) - eta) + eta;
    % One-step forward prediction
    hT1_hat = phi_draw(1) + phi_draw(2)*(phi_draw(1) + phi_draw(2)*h_draw(T-1));
    hf1 = [h_draw(2:end); hT1_hat]; 
    hb1 = [h0_hat; h_draw(1:end-1)];
    
    h_cand = normrnd(h_draw, sigma0_h); % Propose new ht from normal distribution

    % Log ratio of the posterior probability
    logr = log(normpdf(h_cand, [ones(T, 1), hb1] * phi_draw', sqrt(sigmav_draw))) + ...
        log(normpdf(hf1, [ones(T, 1), h_cand] * phi_draw', sqrt(sigmav_draw))) + ...
        log(normpdf(yt - mu_draw, 0, sqrt(exp(h_cand)))) - ...
        log(normpdf(h_draw, [ones(T, 1), hb1] * phi_draw', sqrt(sigmav_draw))) - ...
        log(normpdf(hf1, [ones(T, 1), h_draw] * phi_draw', sqrt(sigmav_draw))) - ...
        log(normpdf(yt - mu_draw, 0, sqrt(exp(h_draw))));

    % Compute acceptance probability
    idxi = log(rand(T, 1)) < logr; 
    h_draw(idxi) = h_cand(idxi);
    
    AccDraws_h = AccDraws_h + nnz(idxi);
    h_sample(:, iter) = h_draw';

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 3) Sample sigma_v conditional on h_t and phi
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    zt = [ones(T - 1, 1), h_draw(1:end-1)];

    SSR = sum((h_draw(2:end) - zt*phi_draw').^2);
    sigmav_draw = 1 / gamrnd(0.5*nu1, 2/(nu0*Sigmav0 + SSR));
    sigmav_sample(iter) = sigmav_draw;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 4) Sample phi1 conditional on h, phi0 and sigmav
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Sample candidate from normal distribution
    phi_hat = ((h_draw(2:end)-phi_draw(1))'*(h_draw(1:end-1)-phi_draw(1)))/...
        sum((h_draw(1:end-1)-phi_draw(1)).^2);
    s_phi = sqrt(sigmav_draw/sum((h_draw(1:end-1)-phi_draw(1)).^2));
    phi1_cand = normrnd(phi_hat,s_phi);

    if abs(phi1_cand) > 1
        % If the candidate is outside of the stationarity region, 
        % we immediately reject the sample
        phi_sample(iter,2) = phi_draw(2);
    else
        % Acceptance-rejection by independence Metropolis-Hastings
        SSR_cand = sum((h_draw(2:end) - zt*[phi_draw(1), phi1_cand]').^2);
        SSR = sum((h_draw(2:end) - zt*phi_draw').^2);

        logr_phi = (rho1-1)*log((1+phi1_cand)/2) + (rho2-1)*log((1-phi1_cand)/2) + ...
            SSR_cand/(2*sigmav_draw) - ...
            (rho1-1)*log((1+phi_draw(2))/2) - (rho2-1)*log((1-phi_draw(2))/2) - ...
            SSR/(2*sigmav_draw);

        accepted = log(rand()) < logr_phi;
        if accepted
            AccDraws_phi = AccDraws_phi + 1;
            phi_draw(2) = phi1_cand;
        end       
        phi_sample(iter,2) = phi_draw(2);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 4) Sample phi0 from normal posterior
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    phi_hat = (sum(h_draw(2:end)-phi_draw(2)*h_draw(1:end-1)) + ...
        a0/A0)/((T-1)+sigmav_draw/A0);
    sig_phi = 1/((T-1)/sigmav_draw + 1/A0);

    phi_draw(1) = normrnd(phi_hat,sqrt(sig_phi));
    phi_sample(iter,1) = phi_draw(1);

end
toc

full_sample = [mu_sample phi_sample sigmav_sample];
burnt_sample = full_sample(1:burnin, :);
full_sample = full_sample(burnin+1:end, :);

%% MCMC Traceplots

figure('position', [355   320   800   400]);
subplot(4, 1, 1);
plot(full_sample(:, 1));
ylabel('\mu');
subplot(4, 1, 2);
plot(full_sample(:, 2))
ylabel('\phi_0');
subplot(4, 1, 3);
plot(full_sample(:, 3))
ylabel('\phi_1');
subplot(4, 1, 4);
plot(full_sample(:, 4));
ylabel('\sigma_v^2');
xlabel('# of iterations');

%% MCMC diagnostics

result = momentg(full_sample)
post_means = [result.pmean]'
post_std = [result.pstd]'
NSE = [result.nse]'
post_median = median(full_sample)'
acceptance_rate = AccDraws_phi/n_sims

%% Plot stochastic volatility
vol_sample = exp(h_sample(:,burnin+1:end));

% 2.5% quantile
lb = quantile(vol_sample, 0.025, 2); 
% 97.5% quantile
ub = quantile(vol_sample, 0.975, 2); 

patch_x = [1:T, T:-1:1];
patch_y = [lb; flip(ub)];

figure('position', [355   320   800   300]);
hold on;
box on;
plot(1:T, GARCH_vol, 'linewidth', 1)
plot(1:T, mean(vol_sample, 2), 'linewidth', 1)
patch(patch_x, patch_y, 'm', 'EdgeColor', 'none', 'FaceAlpha', 0.1);
legend({'GARCH','SV', '95% interquantile'}, 'Location', 'best');
ylims = ylim;
ylabel('S&P500 Stochastic volatility');


idx = find(month(dates) == 1);
idx(month(dates(idx - 1)) == 1) = [];
set(gca, 'xtick', idx);
set(gca, 'XTickLabel', year(dates(idx)));
set(gca, 'xlim', [1, T]);
set(gca, 'ylim', [0, 80]);


%% Error comparison

sv_residuals = (yt-mean(mu_sample))./sqrt(mean(vol_sample,2));
figure('position', [355   320   800   400]);
plotcorrstat(dates, sv_residuals, 30, 1:30)
ylabel('Stochastic volatility');
figure('position', [355   320   800   400]);
normplot(sv_residuals)
ylabel('Stochastic volatility');

figure('position', [355   320   800   400]);
plotcorrstat(dates, residuals, 30, 1:30)
ylabel('GARCH volatility');
figure('position', [355   320   800   400]);
normplot(residuals)
ylabel('GARCH volatility');
