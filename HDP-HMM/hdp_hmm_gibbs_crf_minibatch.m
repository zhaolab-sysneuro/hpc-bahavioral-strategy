function model = hdp_hmm_gibbs_crf_minibatch(x_raw, K_max, numIter, flag_sticky, gamma_input, n_sample)

% seed = 10;
seed = 1;
rng(seed);

%% HDP-HMM / Sticky HDP-HMM Gibbs sampler with CRF table counts
% Each Gibbs iteration fits a randomly sampled continuous subsequence
% from x_raw with length n_sample.
%
% Input:
%   x_raw       : observation sequence, e.g. [0,0,1,2,2,0,0,3]
%   K_max       : finite truncation level
%   numIter     : number of Gibbs sampling iterations
%   flag_sticky : 1 = Sticky HDP-HMM, includes kappa
%                 0 = standard HDP-HMM, kappa fixed to 0
%   gamma_input : if gamma_input > 0, gamma is fixed to this value
%                 if gamma_input <= 0, gamma is optimized by MH
%   n_sample    : length of randomly sampled subsequence in each iteration
%
% Output:
%   model : structure containing inferred parameters and samples
%
% Notes:
%   samples.alpha(iter) records alpha at each iteration.
%   samples.gamma(iter) records gamma at each iteration.
%   samples.start_idx(iter) records the starting index of the sampled segment.

%% -----------------------------
% 1. Preprocess observations
%% -----------------------------

x_raw = x_raw(:)';
T_full = length(x_raw);

if nargin < 6 || isempty(n_sample)
    n_sample = T_full;
end

n_sample = min(n_sample, T_full);

[obsValues, ~, x_full] = unique(x_raw);
x_full = x_full(:)';
V = length(obsValues);

%% -----------------------------
% 2. Hyperparameters
%% -----------------------------

alpha = 5;

if gamma_input > 0
    gamma = gamma_input;
    flag_optimize_gamma = 0;
else
    gamma = 5;
    flag_optimize_gamma = 1;
end

if flag_sticky == 1
    kappa = 20;
else
    kappa = 0;
end

eta = 0.5;

mh_alpha_std = 0.15;
mh_gamma_std = 0.15;
mh_kappa_std = 0.15;

prior.alpha_shape = 2; prior.alpha_rate = 0.1;
prior.gamma_shape = 2; prior.gamma_rate = 0.1;
prior.kappa_shape = 2; prior.kappa_rate = 0.1;

%% -----------------------------
% 3. Initialization
%% -----------------------------

z = randi(K_max, 1, n_sample);

beta = dirichlet_sample(gamma / K_max * ones(1, K_max));

pi_mat = zeros(K_max, K_max);
for j = 1:K_max
    param = alpha * beta;
    if flag_sticky == 1
        param(j) = param(j) + kappa;
    end
    pi_mat(j,:) = dirichlet_sample(param);
end

phi = zeros(K_max, V);
for k = 1:K_max
    phi(k,:) = dirichlet_sample(eta * ones(1, V));
end

samples.alpha = zeros(1, numIter);
samples.gamma = zeros(1, numIter);
samples.kappa = zeros(1, numIter);
samples.K_used = zeros(1, numIter);
samples.loglik = zeros(1, numIter);
samples.start_idx = zeros(1, numIter);
samples.end_idx = zeros(1, numIter);

%% -----------------------------
% 4. Gibbs sampling
%% -----------------------------

hWait = waitbar(0, 'Running HDP-HMM Gibbs Sampling...');
cleanupObj = onCleanup(@() close(hWait));

for iter = 1:numIter

    %% 4.0 Randomly sample a continuous subsequence

    if n_sample < T_full
        start_idx = randi(T_full - n_sample + 1);
        end_idx = start_idx + n_sample - 1;
    else
        start_idx = 1;
        end_idx = T_full;
    end

    x = x_full(start_idx:end_idx);
    T = length(x);

    if length(z) ~= T
        z = randi(K_max, 1, T);
    end

    if mod(iter,10)==0 || iter==1 || iter==numIter
        waitbar(iter / numIter, hWait, ...
            sprintf(['Iteration %d / %d\n' ...
                     'sample = [%d, %d]\n' ...
                     'alpha = %.3f\n' ...
                     'gamma = %.3f\n' ...
                     'kappa = %.3f\n' ...
                     'K_{used} = %d'], ...
                     iter, numIter, start_idx, end_idx, ...
                     alpha, gamma, kappa, length(unique(z))));
    end

    %% 4.1 Sample hidden states using FFBS

    z = sample_z_ffbs(x, pi_mat, phi);

    %% 4.2 Count transitions and emissions based on sampled subsequence

    transCounts = zeros(K_max, K_max);
    emitCounts = zeros(K_max, V);

    for t = 1:T-1
        transCounts(z(t), z(t+1)) = transCounts(z(t), z(t+1)) + 1;
    end

    for t = 1:T
        emitCounts(z(t), x(t)) = emitCounts(z(t), x(t)) + 1;
    end

    %% 4.3 Sample CRF table counts m_jk

    m = sample_crf_table_counts(transCounts, alpha, beta, kappa, flag_sticky);

    %% 4.4 Update beta using CRF table counts

    m_dot_k = sum(m, 1);
    beta_param = gamma / K_max * ones(1, K_max) + m_dot_k;
    beta = dirichlet_sample(beta_param);

    %% 4.5 Update transition matrix pi

    for j = 1:K_max
        param = alpha * beta + transCounts(j,:);

        if flag_sticky == 1
            param(j) = param(j) + kappa;
        end

        pi_mat(j,:) = dirichlet_sample(param);
    end

    %% 4.6 Update emission matrix phi

    for k = 1:K_max
        phi(k,:) = dirichlet_sample(eta * ones(1, V) + emitCounts(k,:));
    end

    %% 4.7 Update alpha by MH

    alpha = mh_update_positive( ...
        alpha, ...
        @(a) logpost_alpha(a, beta, pi_mat, kappa, flag_sticky, prior), ...
        mh_alpha_std);

    %% 4.8 Update gamma by MH only if gamma_input <= 0

    if flag_optimize_gamma == 1
        gamma = mh_update_positive( ...
            gamma, ...
            @(g) logpost_gamma(g, beta, prior), ...
            mh_gamma_std);
    end

    %% 4.9 Update kappa by MH only for Sticky HDP-HMM

    if flag_sticky == 1
        kappa = mh_update_positive( ...
            kappa, ...
            @(kap) logpost_kappa(kap, alpha, beta, pi_mat, prior), ...
            mh_kappa_std);
    else
        kappa = 0;
    end

    %% 4.10 Store samples

    samples.alpha(iter) = alpha;
    samples.gamma(iter) = gamma;
    samples.kappa(iter) = kappa;
    samples.K_used(iter) = length(unique(z(2:end)));
    samples.loglik(iter) = compute_loglik(x, pi_mat, phi, z);
    samples.start_idx(iter) = start_idx;
    samples.end_idx(iter) = end_idx;
    samples.pi_trace(iter,:,:) = pi_mat;

end

%% -----------------------------
% 5. Output
%% -----------------------------

model.z = z;
model.pi = pi_mat;
model.beta = beta;
model.phi = phi;

model.alpha = alpha;
model.gamma = gamma;
model.kappa = kappa;
model.eta = eta;
model.K_max = K_max;

model.flag_sticky = flag_sticky;
model.gamma_input = gamma_input;
model.flag_optimize_gamma = flag_optimize_gamma;

model.obsValues = obsValues;
model.samples = samples;

model.n_sample = n_sample;
model.T_full = T_full;
model.last_sample_idx = [start_idx, end_idx];
model.last_x = x;
model.last_z = z;

model.last_transCounts = transCounts;
model.last_tableCounts = m;
model.last_emitCounts = emitCounts;

model.gamma_trace=samples.gamma;
model.alpha_trace=samples.alpha;

end

%% ============================================================
% Helper functions
%% ============================================================

function m = sample_crf_table_counts(n, alpha, beta, kappa, flag_sticky)

K = size(n, 1);
m = zeros(K, K);

for j = 1:K
    for k = 1:K

        n_jk = n(j,k);

        if n_jk == 0
            m(j,k) = 0;
            continue;
        end

        a_jk = alpha * beta(k);

        if flag_sticky == 1 && j == k
            a_jk = a_jk + kappa;
        end

        count = 0;

        for l = 1:n_jk
            p = a_jk / (a_jk + l - 1);

            if rand < p
                count = count + 1;
            end
        end

        m(j,k) = count;

    end
end

end

function z = sample_z_ffbs(x, pi_mat, phi)

K = size(pi_mat, 1);
T = length(x);

logAlpha = zeros(T, K);
logInit = -log(K) * ones(1, K);

logAlpha(1,:) = logInit + log(phi(:, x(1))' + eps);

for t = 2:T
    for k = 1:K
        tmp = logAlpha(t-1,:) + log(pi_mat(:,k)' + eps);
        logAlpha(t,k) = logsumexp(tmp) + log(phi(k, x(t)) + eps);
    end
end

z = zeros(1, T);

probT = exp(logAlpha(T,:) - logsumexp(logAlpha(T,:)));
z(T) = sample_discrete(probT);

for t = T-1:-1:1
    logProb = logAlpha(t,:) + log(pi_mat(:, z(t+1))' + eps);
    prob = exp(logProb - logsumexp(logProb));
    z(t) = sample_discrete(prob);
end

end

function y = dirichlet_sample(a)

a = max(a, 1e-12);
g = gamrnd(a, 1);
y = g / sum(g);

end

function idx = sample_discrete(p)

p = p / sum(p);
cdf = cumsum(p);
idx = find(rand <= cdf, 1, 'first');

end

function s = logsumexp(a)

m = max(a);
s = m + log(sum(exp(a - m)));

end

function loglik = compute_loglik(x, pi_mat, phi, z)

T = length(x);
loglik = log(phi(z(1), x(1)) + eps);

for t = 2:T
    loglik = loglik ...
        + log(pi_mat(z(t-1), z(t)) + eps) ...
        + log(phi(z(t), x(t)) + eps);
end

end

function newVal = mh_update_positive(oldVal, logpost_fun, proposal_std)

logOld = log(oldVal);
logNew = logOld + proposal_std * randn;

proposal = exp(logNew);

logAccept = logpost_fun(proposal) + logNew ...
          - logpost_fun(oldVal) - logOld;

if log(rand) < logAccept
    newVal = proposal;
else
    newVal = oldVal;
end

end

function lp = logpost_alpha(alpha, beta, pi_mat, kappa, flag_sticky, prior)

K = length(beta);
lp = 0;

for j = 1:K
    param = alpha * beta;

    if flag_sticky == 1
        param(j) = param(j) + kappa;
    end

    lp = lp + log_dirichlet_pdf(pi_mat(j,:), param);
end

lp = lp + log_gamma_prior(alpha, prior.alpha_shape, prior.alpha_rate);

end

function lp = logpost_gamma(gamma, beta, prior)

K = length(beta);
param = gamma / K * ones(1, K);

lp = log_dirichlet_pdf(beta, param);
lp = lp + log_gamma_prior(gamma, prior.gamma_shape, prior.gamma_rate);

end

function lp = logpost_kappa(kappa, alpha, beta, pi_mat, prior)

K = length(beta);
lp = 0;

for j = 1:K
    param = alpha * beta;
    param(j) = param(j) + kappa;
    lp = lp + log_dirichlet_pdf(pi_mat(j,:), param);
end

lp = lp + log_gamma_prior(kappa, prior.kappa_shape, prior.kappa_rate);

end

function lp = log_dirichlet_pdf(x, a)

x = max(x, 1e-12);
a = max(a, 1e-12);

lp = gammaln(sum(a)) - sum(gammaln(a)) + sum((a - 1) .* log(x));

end

function lp = log_gamma_prior(x, shape, rate)

if x <= 0
    lp = -inf;
else
    lp = (shape - 1) * log(x) - rate * x;
end

end
