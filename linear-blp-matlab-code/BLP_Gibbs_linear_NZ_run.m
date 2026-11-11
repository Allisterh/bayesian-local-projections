%BLP_Gibbs_linear_NZ_run.m
%
%Run code to estimate Bayesian Local Projections (BLP) using Gibbs sampling
%with New Zealand data
%
%If you use this BLP method and code, please cite "Diminishing Monetary
%Potency under Fiscal Dominance: A Bayesian Local Projections Analysis" by
%Naveed Javed and James Morley, September 2026, CAMA WP
%
%Code allows for multiple shocks/instruments
%Allows Autocorrelation Correction (AC) using controls, SUR inferences across horizons and possibly endogenous variables
%Allows for BLP-IV estimation using a 2SLS Control Function (CF) approach
%
%Bayesian posterior bands are equal-tailed
%
%By including errors from all equations as controls for serial correlation
%the system becomes SUR (same controls for a given horizon and errors at
%different horizons should be uncorrelated. However, better to be
%parsimonious in controls
%
%Allows for Covid dummies in 2020Q2 and 2020Q3 to handle outliers in real GDP
%Allows for ZLB dummies from 2020Q2 to 2021Q3 to handle ZLB on OCR
%Allow for multiple instruments for OCR
%Need to orthogonalize 1st stage residuals to make 2SLS and 2SLSCF equivalent
%Allows block exogenous controls such as US variables for an SOE
%
%Set to run 500 burn-in and 1500 additional draws. Change to 1000 burn-in/3000 additional draws to exactly replicate paper.
%
% Code last updated by James Morley on 12 October 2026 (general sign restrictions, proxy_first, monthly option and lag-2 serial-correlation diagnostic added November 2026; counterfactual code removed)

clear all
close all
clc

addpath([pwd '/_func/'])

seed=9999;
rng(seed); %fix seed for replicability


%% IMPORT and SETUP DATA
data = xlsread('data_NZ.xlsx','Main');
time = data(:,1);
t1 = find(time==2000.25);
%t1 = find(time==2000.5); %uncomment if using GSS shocks
t2 = find(time==2025.75);


data = data(t1:t2,:);

endogenous_variables = [data(:,2),data(:,4:6)]; %OCR LnGDP Inflation LnTWI(real)
%controls = data(:,x); %Controls to also include on the RHS
controls = 0; %uncomment to not include extra controls

block_exogenous_controls = data(:,10:12); %Block exogenous controls (LWIP, LOIL, LCOMD) that have contemporaneous values included along with the lags
%block_exogenous_controls = 0; %uncomment to not include extra block exogenous controls

shock_series = data(:,8); %Augmented RR shock -- our baseline Romer-Romer type monetary policy shock series
%shock_series = data(:,9); %GSS type monetary policy shocks (pre-scaled by 0.01 in the spreadsheet)
%shock_series = [data(:,8),data(:,9)]; %Two instruments

dummies1 = data(:,17:22); %separate dummies for 2020Q2 to 2021Q3 for ZLB
dummies1 = zeros(length(data),6); %uncomment to remove ZLB dummies
dummies2 = data(:,13:14); %separate dummies for 2020Q2 and 2020Q3 for Covid
dummies2 = zeros(length(data),2); %uncomment to remove Covid dummies

norm_sign = ones(size(shock_series,2),1); %1 normalizes as a contractionary monetary policy shock

shock_series_positive = shock_series.*(shock_series>0);
shock_series_negative = shock_series.*(shock_series<0);
%shock_series = shock_series_positive; %uncomment to use positive shocks only
%shock_series = shock_series_negative; %uncomment to use negative shocks only

sign_asymmetry = false; %set to true to select the two-shock (expansionary vs contractionary) case; false keeps the single symmetric shock series -- matches the toggle in BLP_Gibbs_regimes_NZ_run.m
if sign_asymmetry
    norm_sign = [1;-1]; %multiplies second shock by -1 to normalize as an expansionary shock
    shock_series = [norm_sign(1)*shock_series_positive,norm_sign(2)*shock_series_negative]; %considers both shock series jointly -- if shock_series above holds more than one raw instrument column (e.g. iv=1 with multiple instruments), this stacks ALL of their positive parts first, then ALL of their negative parts, so the contractionary group (first half of columns) and expansionary group (second half) each keep every original instrument, split by its own sign; matches the contiguous-halves instrument grouping BLP_gibbs_linear_core.m/CLP_fcn.m/tslscf_fcn.m use via opts.l when iv=1
end

monthly = 0; %data frequency: 0 = quarterly (as in the NZ application), 1 = monthly. Switches the shape-prior hyperparameters (peak, base, gamma_lr) to their recommended monthly values and the default h_max to 60 months (5 years) -- see GLOBAL SETTINGS and INFORMATIVE PRIORS below. All horizons (h_max, sign_lag, peak) are then in months.

proxy_first = 1; %1 (default) if the shock series is a proxy/instrument for the FIRST response variable in endogenous_variables (here the OCR) -- put the proxied variable first. Set to 0 if the shock series does not proxy for any response variable: basic (unnormalized) LP/BLP only, so normalize and iv must be 0, and the impact prior treats variable 1 like the other response variables

T = length(endogenous_variables);
N = size(endogenous_variables,2); %number of endogenous variables
k=size(shock_series,2); %number of shocks series

iv = 0; %set to 1 to consider LP-IV -- set here, ahead of the other options under GLOBAL SETTINGS below, because l's derivation just below depends on it
%A flat/diffuse prior on the first-stage can cause issues: with sign asymmetry, set at least minnesota_variance=1 below to keep BLP-IV close to LP-IV when other priors are left flat/diffuse -- this works by keeping the first-stage estimates well behaved.
if iv && ~proxy_first
    error('iv=1 requires proxy_first=1 (LP-IV instruments the first response variable with the shock series).');
end
if iv
    if sign_asymmetry
        l = 2; %two instrumented variables (contractionary-OCR, expansionary-OCR) when combining iv=1 with sign_asymmetry -- k's columns are grouped into these two sides in contiguous halves (first half contractionary, second half expansionary, per the sign_asymmetry block above), however many raw instruments went into each half
    else
        l = 1; %always exactly one variable (OCR) instrumented when iv=1 without sign_asymmetry, however many instruments k counts in shock_series -- applied automatically now rather than requiring l=1 to be uncommented by hand
    end
else
    l = k; %number of shocks/instrumented variables (one shock per instrumented variable in the non-IV case)
end


%% GLOBAL SETTINGS
p = 4; %number of lags to include in controls
h_max = 20;   %maximum horizon for IRFs (default 5 years: 20 quarters, or 60 months when monthly=1)
if monthly
    h_max = 60;
end
normalize = 0; %set to 1 to rescale IRFs by initial impact on first outcome variable (e.g., OCR) if not using IV. Does not affect IV. Requires proxy_first=1
if normalize && ~proxy_first
    error('normalize=1 requires proxy_first=1 (IRFs are normalized by the impact on the first, proxied response variable).');
end
c_level = 0.68;%0.95; %coverage for confidence bands
ld = 0; %set to 1 to consider long differences specification
ps = 0; %set to 1 to use Piger-Stockwell controls of first differences of response variable -- requires ld=1 and iv=0 (enforced just below): under ps=1, the response variable's own lag is only entered into the regression when ld=1 (there is no levels alternative, so it is dropped entirely if ld=0), and ps has no effect on the iv=1 control-function branch, so ps=1 with iv~=0 would silently do nothing rather than error
trend = 1; %1 includes a linear time trend with the intercept in every LP regression (as in the paper); 0 drops it (intercept only, as in Piger and Stockwell, 2025, for long differences). Applies to the BLP and to the classical LP comparisons
if ps == 1 && (ld ~= 1 || iv ~= 0)
    error('ps=1 (Piger-Stockwell specification) requires ld=1 and iv=0: with ld=0 the response variable''s own lag is dropped from the regression entirely under ps=1, and with iv~=0 the ps setting has no effect at all (the control-function branch is ps-invariant). Set ld=1 and iv=0, or use ps=0.');
end

prior_only = 0; %set to 1 to simulate every draw from the prior instead of the data-conditioned posterior (GEN_BETA.m/GEN_SIG.m both then ignore the data entirely) -- e.g. to visualize what the informative priors above imply for the IRFs before seeing the data. Requires minnesota_variance=1 above (a proper, non-degenerate prior on the residual variance) -- v0=0,d0=0 otherwise, which is improper and produces NaN/Inf draws here. Set monthly=1 above to see the prior implied by the monthly hyperparameters.
if prior_only
    c_level = 0.99; %matches the paper's own prior-only comparison (BLP_replication_main.m's Figure B-1 uses c_level=0.99) -- overridden here (rather than left at the c_level set above) so every band/interval reported below, in both the LP row and all BLP legends/figures, reflects the wider prior-only coverage rather than the posterior-tuned 0.68 default.
    %c_level = 0.95; %uncomment (and edit the value) to override the 0.99 default above with a different coverage level for the prior-only bands/intervals
end


%% AUTOCORRELATION CORRECTION
neweywest = 1; %set to 1 to use Newey-West standard errors for Classical LP, OLS otherwise
ac = 1; %set to 1 to use univariate controls for serial correlation for BLP, 2 to use lagged errors (or projection errors) for all N
    control_truncation=1;%h_max;  %if set to h_max, no truncation for ac=1. Otherwise, the number of lags given
leads = 1; %set to 1 to control for leads and lags of MP shock or -1 to consider AR(p) correction
if leads==-1
[~,shock_series_resids]=ARp_fcn(shock_series,p,0,0,0);
shock_series=shock_series_resids;
end


%% INFORMATIVE PRIORS FOR BLP ANALYSIS
minnesota_lr = 0; %set to 1 to put Minnesota like prior on IRFs at longer horizons
    lambda_irf = 30;%0.75;

smoothing = 1; %set to 1 to estimate using changes in IRFs to impose smoothing shape prior, overrides minnesota_lr
    lambda_smooth = 8; %set low to smooth IRFs -- kept the same across frequencies below
    if monthly
        %Recommended monthly-frequency hyperparameters, hard-coded (not
        %derived at runtime) since gamma_lr needed a manual rounding
        %judgment call -- see BLP_Gibbs_linear_NZ_prior_slow.m for the logic:
        %peak: 3x the quarterly peak, to land on the same real-time point
        %base: quarterly base scaled by sqrt(3) (solving for base holding
        %      lambda_smooth/base's implied variance comparable after
        %      3x'ing the number of horizons), then rounded (12*sqrt(3)=20.8->20)
        %gamma_lr: quarterly gamma_lr^(1/3) matches its half-life in months
        %      (0.99^(1/3)=0.9967), rounded up a little further so the
        %      5-year prior IRFs look similar in shape across frequencies
        gamma_lr = 0.999;
        peak = 18;
        base = 20;
    else
        gamma_lr = 0.99;         %set low to place prior mean that IRFs are zero in the long run, set to 1 for prior mean for diff is zero
        peak = 6;   %set prior on peak impact period of shock
        base = 12;
    end

minnesota_lags = 0; %set to 1 to put Minnesota like prior on coefficients on lagged variables, can consider larger p
    lambda_lags = 1;%0.2;

minnesota_variance = 1; %set to 1 to put Minnesota like prior on variance of residual
    lambda_var = 50; %9 (50) implies prior standard deviation of about 1 (0.23) if target mode is 1
    horizon_expand = 0; %set to 1 if target mode of variance increases with horizon

minnesota_controls=1; %set to 1 to put Minnesota like prior on controls if AC=1
    lambda_controls=(1/3); %set low to shrink to prior mean over long horizons, 0.25 has more precise long horizon estimates, 0.5 removes all serial correlation
    mean_decay=0.75;       %set to 1 to have prior mean of 1, less than one to decay over horizons

prior_iv=1; %set to 1 to impose a prior on first-stage instrument relationship to endogenous variable
    iv_prior_mean=1;
    iv_prior_var=0.5^2;

impact_prior=1; %set to 1 to impose prior of norm_sign on initial effects of shock on policy variable
    lambda_impact = 0.5; %set shrinkage on impact prior for OCR
    lambda_impact_others=15; %set shrinkage on impact priors for other variables, noting larger is less informative, but implies larger prior mean for IRFs when considering sign restrictions

sign_lag=-1; %restrictions below bind for horizons h=0,...,sign_lag (0 = impact only); set to -1 for no sign restrictions
sign_restrictions = [0;-1;-1;1]; %one entry per endogenous variable, in the column order of endogenous_variables: +1 = response non-negative, -1 = non-positive, 0 = unrestricted. Stated for a shock with norm_sign=+1 (here a contractionary shock) and flipped automatically for norm_sign=-1 (the expansionary side under sign_asymmetry). NZ default: OCR unrestricted, GDP and inflation fall, real TWI appreciates. Set to [] to fall back on the legacy restrictions hard-coded in GEN_BETA.m (identical to this default for the NZ application).


%% Internal/derived settings ("the attic" -- plumbing values used by the code below rather than choices meant to be adjusted)
n_x = 0; %flags "no regime switching" to the shared CLP_fcn/BLP core machinery -- this (non-regime-switching) script always leaves it at 0
orthogonalize = 1; %orthogonalize first-stage errors for the control-function 2SLS estimator -- only has any effect when iv=1 and l>1 (multiple instrumented variables/sides); otherwise inert
p_max = p; %convenience copy of p used wherever a fixed lag count is needed (p itself gets reset to p_max+1 when ld==1 inside the CLP/BLP lag-construction loops)
   

%% Classical LP using OLS or IV
[irfs, ses, ~, ~] = CLP_fcn(endogenous_variables,controls,block_exogenous_controls,shock_series,l,orthogonalize,dummies1,dummies2,norm_sign,iv,h_max,p_max,neweywest,ld,leads,ps,trend);

if normalize == 1  %rescale irfs to be for unit shock in terms of impact on OCR
%rescale irfs to be for unit shock in terms of impact on OCR
%rescale irf SEs given unit shock in terms of OCR, noting SE for contemporaneous impact on OCR is zero
normalize_factor=[];
for i=1:l
    normalize_factor=[normalize_factor;irfs(1,i)];
end
for j=1:N
    for i=1:l
            irfs(:,(j-1)*l+i)=norm_sign(i)*irfs(:,(j-1)*l+i)./normalize_factor(i);
            ses(:,(j-1)*l+i)=ses(:,(j-1)*l+i)./abs(normalize_factor(i));
            ses(1,i)=0;
    end
end
end

%Construct confidence bands
irfs_lower=irfs-norminv(1-(1-c_level)/2)*ses;
irfs_upper=irfs+norminv(1-(1-c_level)/2)*ses;

h_max_ols = h_max; %horizon for the classical LP comparison and its figure panel -- the same h_max as the BLP (kept as a separate name only because the plotting code below uses it)


%% Bayesian LP using Gibbs -- via BLP_gibbs_linear_core.m

N0 = 500;1000;       %NUMBER OF DRAWS TO LEAVE OUT
MM = 1500;3000;       %NUMBER OF DRAWS AFTER BURN-IN

opts = struct();
opts.norm_sign = norm_sign;
opts.n_x       = n_x;
opts.l         = l; %without this, BLP_gibbs_linear_core.m re-derives l as k (one instrument per instrumented variable) and never sees the l=1 override above for the "many instruments, one variable instrumented" IV case -- passing it through here keeps the Bayesian sampler consistent with the l already used for the classical LP call above

opts.controls                 = controls;
opts.block_exogenous_controls = block_exogenous_controls;
opts.dummies1                 = dummies1;
opts.dummies2                 = dummies2;

opts.p     = p;
opts.h_max = h_max;
opts.iv    = iv;
opts.orthogonalize = orthogonalize;
opts.c_level = c_level;
opts.ld    = ld;
opts.ps    = ps;
opts.trend = trend;

opts.neweywest = neweywest;
opts.ac        = ac;
opts.control_truncation = control_truncation;
opts.leads     = leads;

opts.minnesota_lr = minnesota_lr;
opts.lambda_irf   = lambda_irf;

opts.smoothing     = smoothing;
opts.lambda_smooth = lambda_smooth;
opts.gamma_lr      = gamma_lr;
opts.peak          = peak;
opts.base          = base;

opts.minnesota_lags = minnesota_lags;
opts.lambda_lags    = lambda_lags;

opts.minnesota_variance = minnesota_variance;
opts.lambda_var         = lambda_var;
opts.horizon_expand     = horizon_expand;

opts.minnesota_controls = minnesota_controls;
opts.lambda_controls    = lambda_controls;
opts.mean_decay         = mean_decay;

opts.prior_iv      = prior_iv;
opts.iv_prior_mean = iv_prior_mean;
opts.iv_prior_var  = iv_prior_var;

opts.proxy_first          = proxy_first;
opts.impact_prior         = impact_prior;
opts.lambda_impact        = lambda_impact;
opts.lambda_impact_others = lambda_impact_others;

opts.sign_lag = sign_lag;
opts.sign_restrictions = sign_restrictions;

opts.prior_only = prior_only;

opts.N0 = N0;
opts.MM = MM;
opts.verbose = true;

result = BLP_gibbs_linear_core(endogenous_variables, shock_series, opts);

irfmm     = result.irfmm;
irfdiffmm = result.irfdiffmm;
corrmm    = result.corrmm;
corr2mm   = result.corr2mm;
corr5mm   = result.corr5mm;
corrhmm   = result.corrhmm;


if normalize == 1  %rescale irfs to be for unit shock in terms of impact on OCR
normalize_factor=[];
for i=1:l
    normalize_factor=[normalize_factor,mean(irfmm(:,i))]; %normalize with posterior mean to get identical results to OLS given diffuse/improper priors
    %normalize_factor=[normalize_factor,irfmm(:,i)]; %normalize taking first-stage uncertainty into account
end
for h=0:h_max
    for j=1:N
        for i=1:l
            irfmm(:,h*N*l+(j-1)*l+i)=norm_sign(i)*irfmm(:,h*N*l+(j-1)*l+i)./normalize_factor(:,i);
            if smoothing == 1
            irfdiffmm(:,h*N*l+(j-1)*l+i)=norm_sign(i)*irfdiffmm(:,h*N*l+(j-1)*l+i)./normalize_factor(:,i); %irfdiffmm has the same l*N-per-horizon layout as irfmm (l instrumented-variable dimensions in the output regardless of how many raw instrument/shock columns k feeds in), so it must be indexed by l, not k
            end
        end
    end
end
    for i=1:l
    irfmm(:,i)=norm_sign(i).*ones(size(irfmm,1),1);  %uncomment when normalizing all other horizons and variables by posterior mean
    irfdiffmm(:,i)=norm_sign(i).*ones(size(irfmm,1),1); %uncomment when normalizing all other horizons and variables by posterior mean
    end
end

fnl_mat = irfmm;
fnl_mat2 = [];
if smoothing == 1
fnl_mat2 = irfdiffmm;
end

fnl_mat=[fnl_mat,fnl_mat2];

posterior_moments=moments_fcn(fnl_mat,c_level);

%Store BLP results in same format as CLP
Bayes_irfs=[];
Bayes_ses=[];
Bayes_irfs_lower=[];
Bayes_irfs_upper=[];

for h=0:h_max
Bayes_irfs = [Bayes_irfs, posterior_moments(l*N*h+1:l*N*h+l*N,1)];
Bayes_ses = [Bayes_ses, posterior_moments(l*N*h+1:l*N*h+l*N,2)];
Bayes_irfs_lower = [Bayes_irfs_lower, posterior_moments(l*N*h+1:l*N*h+l*N,4)];
Bayes_irfs_upper = [Bayes_irfs_upper, posterior_moments(l*N*h+1:l*N*h+l*N,5)];
end

Bayes_irfs = Bayes_irfs';
Bayes_ses = Bayes_ses';
Bayes_irfs_lower = Bayes_irfs_lower';
Bayes_irfs_upper = Bayes_irfs_upper';

mean(mean(corrhmm));%calculate average absolute value of sample correlations of errors across variables for each horizon
mean(mean(corrmm)); %calculate average first-order serial correlation across all variables and horizon
avg_corrhmm = mean(mean(corrhmm));
avg_corrmm  = mean(mean(corrmm));
avg_corr2mm = mean(mean(corr2mm(:,2*N+1:end))); %second-order serial correlation averaged over all variables and horizons h>=2 (where it can arise from the serial-correlation control)

% Creates output file to store results
[output, fopen_msg] = fopen(fullfile(pwd,'results.txt'),'w');
if output < 0 %e.g., results.txt is open/locked in another program, or the current folder is read-only
    warning('BLP:resultsFile','Could not open results.txt for writing (%s), so the summary results are printed to the Command Window instead.', fopen_msg);
    output = 1; %1 = Command Window
end

% Print table of results into 'results.txt'
fprintf(output, 'Confidence level for equal-tailed bands\n');
fprintf(output, '        %0.8f\n', c_level);

fprintf(output, 'mean, std dev, median, lower percentile, upper percentile\n');
fprintf(output, '        %0.8f        %0.8f        %0.8f        %0.8f         %0.8f\n', posterior_moments(:, 1:5));

fprintf(output, 'correlation between draws(1, 5, 10, 20 draws apart)\n');
fprintf(output, '        %0.8f        %0.8f        %0.8f        %0.8f\n', posterior_moments(:, 6:9));

fprintf(output, 'average absolute value of sample correlations of errors across variables for each horizon\n');
fprintf(output, '        %0.8f\n', avg_corrhmm);

fprintf(output, 'average first-order serial correlation across all variables and horizon\n');
fprintf(output, '        %0.8f\n', avg_corrmm);

fprintf(output, 'average second-order serial correlation across all variables and horizons h>=2\n');
fprintf(output, '        %0.8f\n', avg_corr2mm);

if output > 2
    fclose(output);
end

% Display confidence level
disp('confidence level for equal-tailed bands')
disp(c_level)

% Display parameter estimates
disp('mean ~ std dev ~ median ~ lower percentile ~ upper percentile')
disp(posterior_moments(:, 1:5))

% Display correlation of draws
disp('correlation between draws(1, 5, 10, 20 draws apart)')
disp(posterior_moments(:, 6:9))

% Display average absolute value of sample correlations of errors across variables for each horizon
disp('average absolute value of sample correlations of errors across variables for each horizon')
disp(avg_corrhmm)

% Display average first-order serial correlation across all variables and horizon
disp('average first-order serial correlation across all variables and horizon')
disp(avg_corrmm)

% Display average second-order serial correlation across all variables and horizons h>=2
disp('average second-order serial correlation across all variables and horizons h>=2')
disp(avg_corr2mm)


%% PLOT RESULTS
%White figure background instead of MATLAB's on-screen default light
%gray, and no border box around legends -- matches the paper/replication
%code's own figures: a white background is how these figures get
%exported/printed for the paper, and a boxed legend was never part of
%that look either, so setting these as figure/legend 'default' properties
%makes both the on-screen appearance too.
set(groot,'defaultFigureColor','white');
set(groot,'defaultLegendBox','off');

zz_ols=zeros(1,h_max_ols+1); %LP panel's x-range (same as h_max)
zz=zeros(1,h_max+1);
%When iv=1 (LP-IV/BLP-IV), every "LP"/"BLP" label in the figures below is
%shown as "LP-IV"/"BLP-IV" instead, via these two prefixes, so the panel
%titles, axis-label switch-cases, and the BLP IRF compare figure all stay
%consistent with which estimator actually produced the results.
if iv
    lp_prefix = 'LP-IV';
    blp_prefix = 'BLP-IV';
else
    lp_prefix = 'LP';
    blp_prefix = 'BLP';
end

names1 = {[lp_prefix,' for Cash rate'],
         [lp_prefix,' for GDP'],
         [lp_prefix,' for Inflation'],
         [lp_prefix,' for TWI']};
names2 = {[blp_prefix,' for Cash rate'],
         [blp_prefix,' for GDP'],
         [blp_prefix,' for Inflation'],
         [blp_prefix,' for TWI']};

%Panel font size for the IRF grids below (Figures 1 and 2), matching the
%more legible scale used in the paper's own figures.
irf_panel_fontsize = 12;

%All Bayesian (BLP) legends below use these two labels instead of hardcoded
%"Posterior means"/"credible intervals" text, since prior_only=1 makes
%every BLP draw a simulation from the prior rather than a data-conditioned
%posterior (GEN_BETA.m/GEN_SIG.m ignore the data entirely in that case --
%see the prior_only setting above). The LP rows/figures are unaffected and
%keep their own "Point estimates"/"confidence intervals" wording
%unconditionally, since classical LP is estimated directly from the data
%regardless of prior_only.
if prior_only
    blp_label = 'Prior means';
    blp_interval_label = 'prior intervals';
else
    blp_label = 'Posterior means';
    blp_interval_label = 'credible intervals';
end

figure
set(gcf, 'Units','inches','Position',[1 1 13.333 7.076]); %widescreen, slide-ready size for a multi-panel figure -- matches the paper's own co-author-formatted .fig files, and fixes multi-panel grids being illegibly cramped in MATLAB's default 560x420 figure window
ax_lp = gobjects(1,N);
ax_blp = gobjects(1,N);
for j=1:N
ax_lp(j) = subplot(2,N,j);
h1=plot(0:1:h_max_ols, zz_ols, 'k-');
hold on
grpyat=[(0:1:h_max_ols)', irfs_lower(:,(j-1)*l+1); (h_max_ols:-1:0)' flipud(irfs_upper(:,(j-1)*l+1))];
grpyat2=[(0:1:h_max_ols)', irfs_lower(:,(j-1)*l+l); (h_max_ols:-1:0)' flipud(irfs_upper(:,(j-1)*l+l))];
patch(grpyat2(:,1), grpyat2(:,2), [0.7 0.7 0.7],'edgecolor',[0.7 0.7 0.7]);
alpha(0.4)
hpatch_lp=patch(grpyat(:,1), grpyat(:,2), [0.4 0.4 0.4],'edgecolor',[0.4 0.4 0.4]);
alpha(0.3)
hold on
h2_lp=plot(0:1:h_max_ols, irfs(:,(j-1)*l+l), 'color',[0 0.5 0],'linestyle','-.','LineWidth', 1.4);
hold on
h3_lp=plot(0:1:h_max_ols, irfs(:,(j-1)*l+1), 'r-','LineWidth', 1.4);
hold on
h5=plot(0:1:h_max_ols, zz_ols, 'k-');
axis tight
xlim([0 h_max_ols]); %same horizon as the BLP panel below
set(gca,'FontSize',irf_panel_fontsize)
title(names1{j});

switch names1{j}
    case {char([lp_prefix,' for Cash rate']),char([lp_prefix,' for Inflation'])}
      ylabel('ppts')
   otherwise
      ylabel('log points')
end



end
for j=1:N
ax_blp(j) = subplot(2,N,N+j);
h1=plot(0:1:h_max, zz, 'k-');
hold on
grpyat=[(0:1:h_max)', Bayes_irfs_lower(:,(j-1)*l+1); (h_max:-1:0)' flipud(Bayes_irfs_upper(:,(j-1)*l+1))];
grpyat2=[(0:1:h_max)', Bayes_irfs_lower(:,(j-1)*l+l); (h_max:-1:0)' flipud(Bayes_irfs_upper(:,(j-1)*l+l))];
patch(grpyat2(:,1), grpyat2(:,2), [0.7 0.7 0.7],'edgecolor',[0.7 0.7 0.7]);
alpha(0.4)
hpatch_blp=patch(grpyat(:,1), grpyat(:,2), [0.4 0.4 0.4],'edgecolor',[0.4 0.4 0.4]);
alpha(0.3)
hold on
h3_blp=plot(0:1:h_max, Bayes_irfs(:,(j-1)*l+l), 'color',[0 0.5 0],'linestyle','-.','LineWidth', 1.4);
hold on
h2_blp=plot(0:1:h_max, Bayes_irfs(:,(j-1)*l+1), 'r-','LineWidth', 1.4);
hold on
h5=plot(0:1:h_max, zz, 'k-');
axis tight
xlim([0 h_max]); %equals h_max_ols (matching the LP panel above)
set(gca,'FontSize',irf_panel_fontsize)
title(names2{j});

switch names2{j}
   case {char([blp_prefix,' for Cash rate']),char([blp_prefix,' for Inflation'])}
      ylabel('ppts')
   otherwise
      ylabel('log points')
end

end

if l>1
lgd_lp = legend(ax_lp(N), [h3_lp, h2_lp, hpatch_lp], {'Point estimates given contractionary shock','Point estimates given expansionary shock',[num2str(100*c_level),'% confidence intervals']}, 'Orientation','horizontal');
place_legend_below_row(lgd_lp, ax_lp, 0.035);

lgd_blp = legend(ax_blp(N), [h2_blp, h3_blp, hpatch_blp], {[blp_label,' given contractionary shock'],[blp_label,' given expansionary shock'],[num2str(100*c_level),'% ',blp_interval_label]}, 'Orientation','horizontal');
place_legend_below_row(lgd_blp, ax_blp, 0.035);
else
lgd_lp = legend(ax_lp(N), [h3_lp, hpatch_lp], {'Point estimates',[num2str(100*c_level),'% confidence intervals']}, 'Orientation','horizontal');
place_legend_below_row(lgd_lp, ax_lp, 0.035);

lgd_blp = legend(ax_blp(N), [h2_blp, hpatch_blp], {blp_label,[num2str(100*c_level),'% ',blp_interval_label]}, 'Orientation','horizontal');
place_legend_below_row(lgd_blp, ax_blp, 0.035);
end

fig1_handle = ancestor(ax_lp(1),'figure');
fig1_handle.SizeChangedFcn = @(~,~) place_legend_below_row(lgd_lp, ax_lp, 0.035) && place_legend_below_row(lgd_blp, ax_blp, 0.035);

hold off


posterior_moments_sc=moments_fcn(corrmm,0.95);

%Store results for serial correlation of residuals
sc_pe=[];
sc_sd=[];
sc_lower=[];
sc_upper=[];


for h=0:h_max
sc_pe = [sc_pe, posterior_moments_sc(N*h+1:N*h+N,1)];
sc_sd = [sc_sd, posterior_moments_sc(N*h+1:N*h+N,2)];
sc_lower = [sc_lower, posterior_moments_sc(N*h+1:N*h+N,4)];
sc_upper = [sc_upper, posterior_moments_sc(N*h+1:N*h+N,5)];
end

sc_pe = sc_pe';
sc_sd = sc_sd';
sc_lower = sc_lower';
sc_upper = sc_upper';

posterior_moments_sc2=moments_fcn(corr2mm,0.95);

sc2_pe=[];
sc2_lower=[];
sc2_upper=[];

for h=0:h_max
sc2_pe = [sc2_pe, posterior_moments_sc2(N*h+1:N*h+N,1)];
sc2_lower = [sc2_lower, posterior_moments_sc2(N*h+1:N*h+N,4)];
sc2_upper = [sc2_upper, posterior_moments_sc2(N*h+1:N*h+N,5)];
end

sc2_pe = sc2_pe';
sc2_lower = sc2_lower';
sc2_upper = sc2_upper';

posterior_moments_sc=moments_fcn(corr5mm,0.95);

sc5_pe=[];
sc5_sd=[];
sc5_lower=[];
sc5_upper=[];

for h=0:h_max
sc5_pe = [sc5_pe, posterior_moments_sc(N*h+1:N*h+N,1)];
sc5_sd = [sc5_sd, posterior_moments_sc(N*h+1:N*h+N,2)];
sc5_lower = [sc5_lower, posterior_moments_sc(N*h+1:N*h+N,4)];
sc5_upper = [sc5_upper, posterior_moments_sc(N*h+1:N*h+N,5)];
end

sc5_pe = sc5_pe';
sc5_sd = sc5_sd';
sc5_lower = sc5_lower';
sc5_upper = sc5_upper';

%% PLOT RESULTS FOR SERIAL CORRELATION OF RESIDUALS
zz=zeros(1,h_max+1);
names1 = {'Serial corr. (1st) for Cash rate',
         'Serial corr. (1st) for GDP',
         'Serial corr. (1st) for Inflation',
         'Serial corr. (1st) for TWI'};
names3 = {'Serial corr. (2nd) for Cash rate',
         'Serial corr. (2nd) for GDP',
         'Serial corr. (2nd) for Inflation',
         'Serial corr. (2nd) for TWI'};
names2 = {'Serial corr. (5th) for Cash rate',
         'Serial corr. (5th) for GDP',
         'Serial corr. (5th) for Inflation',
         'Serial corr. (5th) for TWI'};

if ~prior_only
figure
set(gcf, 'Units','inches','Position',[1 1 13.333 7.076]); %widescreen, slide-ready size for a multi-panel figure -- matches the paper's own co-author-formatted .fig files, and fixes multi-panel grids being illegibly cramped in MATLAB's default 560x420 figure window
ax_sc1 = gobjects(1,N);
ax_sc2 = gobjects(1,N);
ax_sc3 = gobjects(1,N);
for j=1:N
axis tight
ylim([-1 1]);
xlim([0 h_max]);
ax_sc1(j) = subplot(3,N,j);
h1=plot(0:1:h_max, zz, 'k-');
hold on
grpyat=[(0:1:h_max)', sc_lower(:,j); (h_max:-1:0)' flipud(sc_upper(:,j))];
hpatch_sc=patch(grpyat(:,1), grpyat(:,2), [0.7 0.7 0.7],'edgecolor',[0.6 0.6 0.6]);
h2_sc=plot(0:1:h_max, sc_pe(:,j), 'b-','LineWidth', 1.4);
hold on
h5=plot(0:1:h_max, zz, 'k-');
ylim([-1 1]);
xlim([0 h_max]);
axis tight
set(gca,'FontSize',irf_panel_fontsize)
title(names1{j});

end
for j=1:N
axis tight
xlim([0 h_max]);
ylim([-1 1]);
ax_sc3(j) = subplot(3,N,N+j);
h1=plot(0:1:h_max, zz, 'k-');
hold on
grpyat=[(0:1:h_max)', sc2_lower(:,j); (h_max:-1:0)' flipud(sc2_upper(:,j))];
patch(grpyat(:,1), grpyat(:,2), [0.7 0.7 0.7],'edgecolor',[0.6 0.6 0.6]);
plot(0:1:h_max, sc2_pe(:,j), 'b-','LineWidth', 1.4);
hold on
h5=plot(0:1:h_max, zz, 'k-');
xlim([0 h_max]);
ylim([-1 1]);
set(gca,'FontSize',irf_panel_fontsize)
title(names3{j});

end
for j=1:N
axis tight
xlim([0 h_max]);
ylim([-1 1]);
ax_sc2(j) = subplot(3,N,2*N+j);
h1=plot(0:1:h_max, zz, 'k-');
hold on
grpyat=[(0:1:h_max)', sc5_lower(:,j); (h_max:-1:0)' flipud(sc5_upper(:,j))];
patch(grpyat(:,1), grpyat(:,2), [0.7 0.7 0.7],'edgecolor',[0.6 0.6 0.6]);
h2=plot(0:1:h_max, sc5_pe(:,j), 'b-','LineWidth', 1.4);
hold on
h5=plot(0:1:h_max, zz, 'k-');
xlim([0 h_max]);
ylim([-1 1]);
set(gca,'FontSize',irf_panel_fontsize)
title(names2{j});

end

lgd_sc = legend(ax_sc2(N), [h2_sc, hpatch_sc], {blp_label,['95% ',blp_interval_label]}, 'Orientation','horizontal');
place_legend_below_row(lgd_sc, ax_sc2, 0.035);

fig2_handle = ancestor(ax_sc2(1),'figure');
fig2_handle.SizeChangedFcn = @(~,~) place_legend_below_row(lgd_sc, ax_sc2, 0.035);
end


%% For multiple shocks, calculate difference in IRFs for same unit shock
%normalized to be positive and comparison to first shock
if l>1
irfmm_compare=[];
for h=0:h_max
    for j=1:N
        for i=2:l
            irfmm_compare=[irfmm_compare,norm_sign(1)*irfmm(:,h*N*l+(j-1)*l+1)-norm_sign(i)*irfmm(:,h*N*l+(j-1)*l+i)]; %irfmm is indexed by l (the number of instrumented variables/sides), not k (the number of raw instrument/shock columns)
        end
    end
end


posterior_moments_compare=moments_fcn(irfmm_compare,c_level);


%Store BLP results in same format as CLP
Bayes_irfs_compare=[];
Bayes_ses_compare=[];
Bayes_irfs_lower_compare=[];
Bayes_irfs_upper_compare=[];

for h=0:h_max
Bayes_irfs_compare = [Bayes_irfs_compare, posterior_moments_compare((l-1)*N*h+1:(l-1)*N*h+(l-1)*N,1)];
Bayes_ses_compare = [Bayes_ses_compare, posterior_moments_compare((l-1)*N*h+1:(l-1)*N*h+(l-1)*N,2)];
Bayes_irfs_lower_compare = [Bayes_irfs_lower_compare, posterior_moments_compare((l-1)*N*h+1:(l-1)*N*h+(l-1)*N,4)];
Bayes_irfs_upper_compare = [Bayes_irfs_upper_compare, posterior_moments_compare((l-1)*N*h+1:(l-1)*N*h+(l-1)*N,5)];
end

Bayes_irfs_compare = Bayes_irfs_compare';
Bayes_ses_compare = Bayes_ses_compare';
Bayes_irfs_lower_compare = Bayes_irfs_lower_compare';
Bayes_irfs_upper_compare = Bayes_irfs_upper_compare';


zz=zeros(1,h_max+1);
names1 = {[blp_prefix,' IRF compare for Cash rate'],
         [blp_prefix,' IRF compare for GDP'],
         [blp_prefix,' IRF compare for Inflation'],
         [blp_prefix,' IRF compare for TWI']};
%Two-line subtitle (matches the replication code's plot_blp_compare_row/
%paper Figure 12) rather than one long line, so it doesn't spill into the
%neighboring panel in this figure's tight 4-column layout.
subtitle_cmp = {'Contractionary v.','Expansionary MP shock'};

figure
set(gcf, 'Units','inches','Position',[1 1 13.333 7.076]); %widescreen, slide-ready size for a multi-panel figure -- matches the paper's own co-author-formatted .fig files, and fixes multi-panel grids being illegibly cramped in MATLAB's default 560x420 figure window
ax_cmp = gobjects(1,N);
for j=1:N
axis tight
xlim([0 h_max]);
ax_cmp(j) = subplot(1,N,j);
h1=plot(0:1:h_max, zz, 'k-');
hold on
grpyat=[(0:1:h_max)', Bayes_irfs_lower_compare(:,(j-1)*(l-1)+1); (h_max:-1:0)' flipud(Bayes_irfs_upper_compare(:,(j-1)*(l-1)+1))];
hpatch_cmp=patch(grpyat(:,1), grpyat(:,2), [0.7 0.7 0.7],'edgecolor',[0.6 0.6 0.6]);
alpha(0.3)
hold on
h2_cmp=plot(0:1:h_max, Bayes_irfs_compare(:,(j-1)*(l-1)+1), 'b-','LineWidth', 1.4);
hold on
h5=plot(0:1:h_max, zz, 'k-');
xlim([0 h_max]);
axis tight
th_cmp = title(names1{j},subtitle_cmp);
set(th_cmp,'FontSize',12); %main title kept clearly bigger than the subtitle below -- title() returns only the main Title handle; the Subtitle is a separate object (get(gca,'Subtitle')) that does NOT auto-scale off the Title's FontSize, so it has to be set explicitly too, on the next line, or it's left at its own larger default and ends up looking bigger than the title above it
set(get(gca,'Subtitle'),'FontSize',9); %smaller subtitle font, combined with subtitle_cmp's own two-line wrapping above, so this stays inside its own panel instead of spilling into the next one

switch names1{j}
    case {char([blp_prefix,' IRF compare for Cash rate']),char([blp_prefix,' IRF compare for Inflation'])}
      ylabel('ppts')
   otherwise
      ylabel('log points')
end


end

lgd_cmp = legend(ax_cmp(N), [h2_cmp, hpatch_cmp], {blp_label,[num2str(100*c_level),'% ',blp_interval_label]}, 'Orientation','horizontal');
place_legend_below_row(lgd_cmp, ax_cmp, 0.035);

fig_cmp_handle = ancestor(ax_cmp(1),'figure');
fig_cmp_handle.SizeChangedFcn = @(~,~) place_legend_below_row(lgd_cmp, ax_cmp, 0.035);

hold off

end
