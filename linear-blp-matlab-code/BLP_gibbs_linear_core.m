function out = BLP_gibbs_linear_core(endogenous_variables, shock_series, opts)
%BLP_GIBBS_LINEAR_CORE  Function-ized Gibbs-sampling BLP core, refactored from
%   BLP_Gibbs_linear_NZ_slow.m (Javed and Morley, "Diminishing Monetary Potency
%   under Fiscal Dominance").
%
%   out = BLP_GIBBS_CORE(endogenous_variables, shock_series, opts)
%
%   WHAT THIS IS: the sampling loop (Preliminaries through the end of the
%   main Gibbs loop, plus the posterior-moments step) from
%   BLP_Gibbs_linear_NZ_slow.m, cut out and wrapped in a function so it can be
%   called repeatedly inside a Monte Carlo without reloading NZ data each
%   time. Every line of actual sampling logic (the h/i/g loop structure,
%   the construction of x and y at each horizon, the prior setup, the
%   GEN_SIG/GEN_BETA calls, the smoothing-prior bookkeeping) is copied
%   unchanged from the original script. Settings that were hardcoded at
%   the top of the original script (p, h_max, iv, ld, ps, ac, leads, all
%   the prior toggles) are now fields of `opts`, defaulting to the same
%   values used in BLP_Gibbs_linear_NZ_slow.m so behavior is unchanged if you
%   call this with N=4 NZ-shaped inputs and default opts.
%
%   Inputs
%     endogenous_variables : T x N matrix (N=1 for the univariate AR(8)
%                             Monte Carlo)
%     shock_series          : T x k matrix of observed shock(s)
%     opts                  : struct of settings; see DEFAULTS below.
%                             Any field you omit takes the value used in
%                             BLP_Gibbs_linear_NZ.m's global settings.
%
%   Output (struct)
%     out.irfmm            : MM x (l*N*(h_max+1)) matrix of post-burn-in
%                             IRF draws, same layout as the original
%                             script's irfmm
%     out.irfdiffmm         : same, for the change-in-IRF draws, if
%                             opts.smoothing==1 (empty otherwise)
%     out.Bayes_irfs        : (h_max+1) x (l*N) posterior mean IRF
%     out.Bayes_irfs_lower  : (h_max+1) x (l*N) lower c_level credible bound
%     out.Bayes_irfs_upper  : (h_max+1) x (l*N) upper c_level credible bound
%     out.Bayes_irfs_median : (h_max+1) x (l*N) posterior median IRF
%                             (added here; the original script only kept
%                             mean/lower/upper in Bayes_irfs, but the
%                             median is in posterior_moments column 3
%                             already computed by moments_fcn, just not
%                             carried through to Bayes_irfs in the
%                             original)
%     out.corrmm/corr2mm/corr5mm : MM x N matrices of 1st-, 2nd- and 5th-order
%                             residual serial correlation per draw, and
%     out.corrhmm           : MM x (h_max+1) matrix of average absolute
%                             cross-equation residual correlation per
%                             draw at each horizon -- all computed inside
%                             the loop in every version of the original
%                             script, now surfaced so a wrapper can
%                             reproduce its serial-correlation diagnostic
%                             plots

%% Defaults -- same values as the global settings in BLP_Gibbs_linear_NZ.m
controls                  = getopt(opts,'controls',0);
block_exogenous_controls  = getopt(opts,'block_exogenous_controls',0);
dummies1                  = getopt(opts,'dummies1',zeros(size(endogenous_variables,1),1));
dummies2                  = getopt(opts,'dummies2',zeros(size(endogenous_variables,1),1));
norm_sign                 = getopt(opts,'norm_sign',ones(size(shock_series,2),1));
n_x                       = getopt(opts,'n_x',0);

p_max                     = getopt(opts,'p',4);
h_max                     = getopt(opts,'h_max',20);
iv                        = getopt(opts,'iv',0);
orthogonalize             = getopt(opts,'orthogonalize',1);
c_level                   = getopt(opts,'c_level',0.68);
ld                        = getopt(opts,'ld',0);
ps                        = getopt(opts,'ps',0);
trend                     = getopt(opts,'trend',1); %1 (default) includes a linear time trend with the intercept in every LP regression; 0 drops it (intercept only, as in Piger and Stockwell, 2025, for long differences)
if ps == 1 && (ld ~= 1 || iv ~= 0)
    error('ps=1 (Piger-Stockwell specification) requires ld=1 and iv=0 -- see the ps note where these options are set in the calling script. Set opts.ld=1 and opts.iv=0, or opts.ps=0.');
end

neweywest                 = getopt(opts,'neweywest',1); % unused when iv==0,smoothing/Gibbs path, kept for CLP_fcn calls
ac                        = getopt(opts,'ac',1);
control_truncation        = getopt(opts,'control_truncation',1);
leads                     = getopt(opts,'leads',1);

minnesota_lr              = getopt(opts,'minnesota_lr',0);
lambda_irf                = getopt(opts,'lambda_irf',30);

smoothing                 = getopt(opts,'smoothing',1);
lambda_smooth             = getopt(opts,'lambda_smooth',8);
gamma_lr                  = getopt(opts,'gamma_lr',0.99);
peak                      = getopt(opts,'peak',6);
base                      = getopt(opts,'base',12);

minnesota_lags            = getopt(opts,'minnesota_lags',0);
lambda_lags               = getopt(opts,'lambda_lags',1);

minnesota_variance        = getopt(opts,'minnesota_variance',1);
lambda_var                = getopt(opts,'lambda_var',50);
horizon_expand            = getopt(opts,'horizon_expand',0);

minnesota_controls        = getopt(opts,'minnesota_controls',1);
lambda_controls           = getopt(opts,'lambda_controls',1/3);
mean_decay                = getopt(opts,'mean_decay',0.75);

prior_iv                  = getopt(opts,'prior_iv',1);
iv_prior_mean             = getopt(opts,'iv_prior_mean',1);
iv_prior_var              = getopt(opts,'iv_prior_var',0.5^2);

proxy_first               = getopt(opts,'proxy_first',1); %1 (default) = the shock series is a proxy/instrument for the FIRST endogenous variable (e.g., the policy rate), so the impact prior centres that variable's impact response on norm_sign with lambda_impact, and normalization/IV are available; 0 = no proxied variable, so variable 1 gets the same weak impact prior (mean 0, lambda_impact_others) as all other variables and iv must be 0
impact_prior              = getopt(opts,'impact_prior',0);
lambda_impact             = getopt(opts,'lambda_impact',0.5);
lambda_impact_others      = getopt(opts,'lambda_impact_others',15);

sign_lag                  = getopt(opts,'sign_lag',-1);
sign_restrictions         = getopt(opts,'sign_restrictions',[]); %N-by-1 vector, one entry per endogenous variable (in the column order of endogenous_variables): +1 = response must be non-negative, -1 = non-positive, 0 = unrestricted, for horizons h<=sign_lag. Signs are stated for a shock with norm_sign=+1 (e.g., a contractionary shock) and flip automatically for a shock with norm_sign=-1 (e.g., the expansionary side under sign_asymmetry). Leave empty to use the legacy NZ restrictions hard-coded in GEN_BETA.m (GDP and inflation fall, TWI appreciates, OCR unrestricted).

prior_only                = getopt(opts,'prior_only',false); %true simulates every draw from the prior (GEN_BETA/GEN_SIG both ignore the data) instead of the data-conditioned posterior -- see GEN_BETA.m/GEN_SIG.m. Requires minnesota_variance==1 (a proper, non-degenerate prior on the residual variance); v0=0,d0=0 (the minnesota_variance==0 default) is an improper prior and will produce NaN/Inf draws here.

N0                        = getopt(opts,'N0',1000);
MM                        = getopt(opts,'MM',3000);

verbose                   = getopt(opts,'verbose',false);
print_every               = getopt(opts,'print_every',500);
show_progress             = getopt(opts,'show_progress',true); %independent of verbose (which also gates other dumps in the *_replication.m scripts) -- always prints Gibbs-draw progress to the command window every print_every draws, diary-toggled below so it's never written into replication_results.txt while a diary log is active

CAPN = N0 + MM;

if prior_only && minnesota_variance == 0
    warning('BLP_gibbs_linear_core:improperPriorOnVariance', ...
        'opts.prior_only=true with opts.minnesota_variance=0 uses an improper (v0=0,d0=0) prior on the residual variance -- GEN_SIG will produce NaN/Inf draws. Set opts.minnesota_variance=1 for a proper prior when simulating from the prior.');
end

%% Derived sizes
T = length(endogenous_variables);
N = size(endogenous_variables,2);
if proxy_first~=1 && iv==1
    error('iv=1 requires proxy_first=1: LP-IV instruments the first endogenous variable with the shock series.');
end
if ~isempty(sign_restrictions)
    sign_restrictions = sign_restrictions(:);
    if numel(sign_restrictions)~=N || any(~ismember(sign_restrictions,[-1 0 1]))
        error('opts.sign_restrictions must have one entry per endogenous variable (N=%d), each equal to -1, 0, or 1.',N);
    end
end
k = size(shock_series,2);
l = getopt(opts,'l',k); %number of instrumented variables -- defaults to k (one instrument per instrumented variable) unless opts.l overrides it, e.g. opts.l=1 for a "many instruments (k>1), one variable instrumented via control-function 2SLS" case. Must be threaded through from opts rather than re-derived as k unconditionally, so a run/case script's own local l override reaches this function.

group_size = k/l; %number of raw shock_series/instrument columns grouped per instrumented variable ("side") -- 1 in the original designs (k==l: either l=k with one shock per side in the non-IV case, or l=1 with all k instruments jointly serving the single IV side); >1 when opts.l puts fewer sides than instrument columns, e.g. l=2 for the iv=1 & sign_asymmetry=true case with several instruments per sign. Assumes contiguous equal-sized groups: shock_series/z columns (m-1)*group_size+1 : m*group_size belong to side m (matches how a run/case script stacks [positive-part instruments, negative-part instruments] when both iv and sign_asymmetry are on).
if l>1
    norm_sign_z = repelem(norm_sign(1:l),group_size,1); %expands the per-side sign (length l) to one entry per raw instrument column (length k) for the IV prior mean below, which is indexed by instrument column rather than by side
else
    norm_sign_z = norm_sign; %l==1: a single side uses all k instruments jointly with no grouping, so norm_sign (length k, as opts.norm_sign already provides or defaults to) is used directly, unchanged from prior behavior
end

%% Classical LP for initial values (unnormalized point estimates only)
[~, ~, prmtr_point_store, ~] = CLP_fcn(endogenous_variables,controls,block_exogenous_controls,shock_series,l,orthogonalize,dummies1,dummies2,norm_sign,iv,h_max,p_max,neweywest,ld,leads,ps,trend);

if smoothing == 1
    [~, ~, prmtr_point_store, ~] = CLP_diff_fcn(endogenous_variables,controls,block_exogenous_controls,shock_series,l,orthogonalize,dummies1,dummies2,norm_sign,iv,h_max,p_max,neweywest,ld,leads,ps,trend);
end

if iv == 1
    [~, ~, ~, b1] = CLP_fcn(endogenous_variables,controls,block_exogenous_controls,shock_series,l,orthogonalize,dummies1,dummies2,norm_sign,iv,h_max,p_max,neweywest,ld,leads,ps,trend); %ps only affects the iv==0 branch of CLP_fcn, so this call's result (used only for the iv==1 first-stage seed b1) is unaffected by which ps is passed here
    prmtr1_point_store = b1(:,1:l)';
end

[sige2,~] = ARp_fcn(endogenous_variables,4,dummies1,dummies2,ld);

%% Storage
p = p_max;

prmtr_lag_store = zeros(N*(h_max+1),size(prmtr_point_store,2));
if iv == 1
    prmtr_lag_store = [prmtr_lag_store,zeros(N*(h_max+1),l)];
    prmtr1_lag_store = zeros(l,size(prmtr1_point_store,2));
end
if ac==1
    prmtr_lag_store = [prmtr_lag_store,zeros(N*(h_max+1),control_truncation)];
end
if ac==2
    prmtr_lag_store = [prmtr_lag_store,zeros(N*(h_max+1),1)];
end
if smoothing==1
    prmtr_level_store = prmtr_lag_store;
end
if leads==1
    leads_lag_store = zeros(N*(h_max+1)*k,p);
end
has_dummies1 = sum(sum(dummies1))>0;
has_dummies2 = sum(sum(dummies2))>0;

if has_dummies1
    dummies1_lag_store = zeros(h_max+1,size(dummies1,2));
end
if has_dummies2
    dummies2_lag_store = zeros(h_max+1,size(dummies2,2));
end

irfmm   = zeros(MM, l*N*(h_max+1));
corrmm  = zeros(MM, N*(h_max+1));
corr5mm = zeros(MM, N*(h_max+1));
corr2mm = zeros(MM, N*(h_max+1));
corrhmm = zeros(MM, h_max+1);
if smoothing == 1
    irfdiffmm = zeros(MM, k*N*(h_max+1));
end
if ac == 2
    errors_g = zeros(T-(ld==1),N*h_max);
end

%% Precompute the parts of x (and of y) that do not vary across draws
%
%  For a given horizon h and variable i, the regressor matrix x is built
%  from three pieces, in this column order:
%    [ x_base{h+1,i}   -- shock/constant/trend/block-exog/endog-lags/
%                          controls/lagged-shock terms: built purely from
%                          the input data, identical for every draw g
%      <AC-control columns>  -- built from errors_g, genuinely differ by
%                                draw since they depend on this draw's own
%                                lower-horizon residuals
%      x_tail{h+1,i}   -- leads of the shock + i-specific dummy columns:
%                          again built purely from the input data, same
%                          for every draw g ]
%  and y is, before the smoothing adjustment (which IS draw-dependent),
%  also the same for every draw: y_base{h+1,i}.
precompute_x    = (iv==0 && ps==0);
precompute_x_iv = (iv==1 && ps==0);
precompute_x_ps = (iv==0 && ps==1);
if precompute_x
    p_precomp = p_max + (ld==1); % matches what p would be at this point below

    x_base    = cell(h_max+1, N);
    x_tail    = cell(h_max+1, N);
    y_base    = cell(h_max+1, N);
    b_precomp = zeros(h_max+1, N);

    % b0_v/b0_m (the GEN_BETA prior mean/covariance) are rebuilt from
    % scratch -- including a dense n-by-n eye(n) -- on every one of the
    % (h_max+1)*N*CAPN calls to GEN_BETA below, even though (see the
    % PRIOR PARAMETERS section inside the loop) every piece of that
    % construction depends only on (h,i) and on fixed hyperparameters:
    % sige2 is computed once by ARp_fcn before this loop even starts, and
    % every lambda_*/mean_decay/norm_sign value is a fixed input, never
    % updated during Gibbs. The ONE exception is the smoothing==1 case,
    % which layers in a small l-column adjustment based on the PREVIOUS
    % draw's level estimate (prmtr_level_store) -- that piece genuinely
    % changes every draw and is still applied fresh inside the loop.
    % Everything else is computed once here instead of (h_max+1)*N*CAPN
    % times -- same values, same order of overwrites as the inline
    % version (see the comments where it's used below).
    b0_v_base = cell(h_max+1, N);
    b0_m_base = cell(h_max+1, N);

    for hh = 0:h_max
        for ii = 1:N

            pp = p_precomp;

            if ld == 1
                y_base{hh+1,ii} = endogenous_variables(pp+1+hh:T,ii)-endogenous_variables(pp+1-1:T-hh-1,ii);
            else
                y_base{hh+1,ii} = endogenous_variables(pp+1+hh:T,ii);
            end

            xb=[shock_series(pp+1:T-hh,1:k),ones(T-pp-hh,1),repmat((1:1:T-pp-hh)',1,trend)];
            if block_exogenous_controls ~= 0
                xb=[xb,block_exogenous_controls(pp+1:T-hh,:)];
                if ld ==1 && ps == 0
                    xb(:,size(xb,2)-size(block_exogenous_controls,2)+1:size(xb,2))=block_exogenous_controls(pp+1:T-hh,:)-block_exogenous_controls(pp:T-hh-1,:);
                end
            end
            jj=1;
            while jj <= p_max
                xb=[xb,endogenous_variables(pp+1-jj:T-hh-jj,:)];  % ps==0 branch of the original's "if ps==0" append
                if ld == 1
                    xb(:,size(xb,2)-N+1:size(xb,2))=endogenous_variables(pp+1-jj:T-hh-jj,:)-endogenous_variables(pp-jj:T-hh-1-jj,:); % ps==0 branch of the original's ld==1 in-place overwrite
                end
                if controls ~= 0
                    xb=[xb,controls(pp+1-jj:T-hh-jj,:)];
                    if ld == 1 && ps == 0
                        xb(:,size(xb,2)-size(controls,2)+1:size(xb,2))=controls(pp+1-jj:T-hh-jj,:)-controls(pp-jj:T-hh-1-jj,:);
                    end
                end
                if block_exogenous_controls ~= 0
                    xb=[xb,block_exogenous_controls(pp+1-jj:T-hh-jj,:)];
                    if ld == 1 && ps == 0
                        xb(:,size(xb,2)-size(block_exogenous_controls,2)+1:size(xb,2))=block_exogenous_controls(pp+1-jj:T-hh-jj,:)-block_exogenous_controls(pp-jj:T-hh-1-jj,:);
                    end
                end
                if leads==1
                    xb=[xb,shock_series(pp+1-jj:T-hh-jj,:)];
                end
                jj=jj+1;
            end

            x_base{hh+1,ii}    = xb;
            b_precomp(hh+1,ii) = size(xb,2);

            xt=[];
            if leads==1
                jj=1;
                while jj <= min(p_max,hh)
                    xt=[xt,shock_series(pp+1+jj:T-hh+jj,:)];
                    jj=jj+1;
                end
            end
            if has_dummies1 && ii==1
                xt=[xt,dummies1(pp+1+hh:T,:)];
            end
            if has_dummies2 && ii==2
                xt=[xt,dummies2(pp+1+hh:T,:)];
            end
            x_tail{hh+1,ii} = xt;

            if ac == 1
                if hh>0
                    acj0 = 0;
                    if hh>control_truncation
                        acj0 = hh-control_truncation;
                    end
                    ac_width = hh - acj0;
                else
                    ac_width = 0;
                end
            elseif ac == 2
                if hh>0
                    ac_width = N;
                else
                    ac_width = 0;
                end
            else
                ac_width = 0;
            end

            bb   = b_precomp(hh+1,ii);
            n_bh = bb + ac_width + size(xt,2);

            b0_m_bh = zeros(n_bh,1);
            b0_v_bh = 10000000000*eye(n_bh);

            if minnesota_lr==1
                b0_v_bh(1:l,1:l) = (sige2(ii)*(lambda_irf/(1+hh))^2)*eye(l);
            end

            if minnesota_lags==1
                acj=1;
                while acj<=p_max
                    ack=1;
                    while ack<=N
                        b0_v_bh(l+1+trend+(acj-1)*N+ack,l+1+trend+(acj-1)*N+ack)=(sige2(ii)/sige2(ack))*(lambda_lags/acj)^2;
                        if acj==1 && ack==ii
                            b0_m_bh(l+1+trend+(acj-1)*N+ack)=1;
                        end
                        ack=ack+1;
                    end
                    acj=acj+1;
                end
            end

            if ac == 1
                if minnesota_controls == 1
                    if hh>0
                        acj=0; acjj=0;
                        if hh>control_truncation
                            acjj=hh-control_truncation;
                        end
                        while acjj<=hh-1
                            b0_m_bh(bb+1+acj)=mean_decay^(hh-1-acjj);
                            b0_v_bh(bb+1+acj,bb+1+acj)=(lambda_controls/(hh-acjj))^2;
                            acj=acj+1;
                            acjj=acjj+1;
                        end
                    end
                end
            end

            if ac == 2
                if minnesota_controls == 1
                    if hh>0
                        b0_m_bh(bb+1:bb+N)=zeros(N,1); b0_m_bh(bb+ii)=1;
                        b0_v_bh(bb+1:bb+N,bb+1:bb+N)=(lambda_controls^2)*eye(N);
                    end
                end
            end

            if impact_prior==1
                if hh==0
                    if ii==1 && proxy_first==1
                        b0_m_bh(1:l)=norm_sign(1:l);
                        b0_v_bh(1:l,1:l)=sige2(ii)*(lambda_impact^2)*eye(l);
                    else
                        b0_v_bh(1:l,1:l)=sige2(ii)*(lambda_impact_others^2)*eye(l);
                    end
                end
            end

            b0_v_base{hh+1,ii} = b0_v_bh;
            b0_m_base{hh+1,ii} = b0_m_bh;

        end
    end
end

%% Precompute for iv==1 & ps==0
%
%  For iv==1, the second-stage regressor x_hat is [x, AC-control columns]
%  where x = [instrumented_variables(window), const, trend,
%  block-exog?, endog-lags, controls?, block-exog-lags?] followed by an
%  l-column placeholder that gets overwritten every draw with v (the
%  first-stage IV residuals -- the iv==1 analogue of the AC columns:
%  genuinely draw-dependent, since it depends on this draw's own
%  first-stage betag1). Everything else here -- instrumented_variables,
%  x itself, z (built from x), and z11 (built from z, and never touched
%  by the i-specific dummy columns that x_hat/z get) -- is built purely
%  from fixed inputs (endogenous_variables, shock_series, norm_sign,
%  block_exogenous_controls, controls, dummies1/2), so it is exactly as
%  draw-invariant as x_base is for the non-iv case. This only covers
%  ps==0 (see the note above precompute_x_ps below); iv==1 & ps==1 falls
%  back to the original inline code.
%
%  The first-stage IV regression itself (the i==1,h==0 special case
%  further down, drawing betag1 fresh from x/z) is left untouched -- it
%  only runs once per draw (not once per (h,i,draw)), so it is nowhere
%  near the volume that makes the second-stage regressor construction
%  worth precomputing.
if precompute_x_iv
    p_precomp_iv = p_max + (ld==1); % matches what p would be at this point below

    % instrumented_variables does not depend on h or i at all (only on
    % endogenous_variables/shock_series/norm_sign, all fixed), so unlike
    % everything else here it needs no (h,i) cell array -- just compute it
    % once. The inline iv==1 branch below still recomputes it itself
    % (N*(h_max+1) times per draw); this fixed copy is only used by the
    % fast path.
    instrumented_variables_iv_fixed = norm_sign(1)*endogenous_variables(:,1);
    if l>1
        instrumented_variables_iv_fixed = [];
        m=1;
        while m<=l
            grp = (m-1)*group_size+1:m*group_size; %side m's own group of instrument columns within shock_series (contiguous, group_size wide -- see group_size's definition above)
            instrumented_variables_iv_fixed = [instrumented_variables_iv_fixed,norm_sign(m)*endogenous_variables(:,1).*any(shock_series(:,grp)>0,2)]; %any(...) unions across a side's instrument columns since they need not all be nonzero at exactly the same times
            m=m+1;
        end
    end

    x_base_iv    = cell(h_max+1, N); % includes the trailing l zero
                                      % placeholder columns for v
    z_base_iv    = cell(h_max+1, N); % z before any dummy columns
    z11_base_iv  = cell(h_max+1, N); % fully fixed -- no runtime rebuild needed
    x_tail_iv    = cell(h_max+1, N); % leads then i-specific dummies (x_hat only)
    z_tail_iv    = cell(h_max+1, N); % i-specific dummies only (z gets no leads)
    y_base_iv    = cell(h_max+1, N);
    b_precomp_iv = zeros(h_max+1, N);

    b0_v_base_iv = cell(h_max+1, N);
    b0_m_base_iv = cell(h_max+1, N);

    for hh = 0:h_max
        for ii = 1:N

            pp = p_precomp_iv;

            if ld == 1
                y_base_iv{hh+1,ii} = endogenous_variables(pp+1+hh:T,ii)-endogenous_variables(pp+1-1:T-hh-1,ii);
            else
                y_base_iv{hh+1,ii} = endogenous_variables(pp+1+hh:T,ii);
            end

            xb=[instrumented_variables_iv_fixed(pp+1:T-hh,:)-instrumented_variables_iv_fixed(pp:T-hh-1,:),ones(T-pp-hh,1),repmat((1:1:T-pp-hh)',1,trend)];
            if block_exogenous_controls ~= 0
                xb=[xb,block_exogenous_controls(pp+1:T-hh,:)];
                if ld == 1   % ps==0 forced by the precompute_x_iv guard
                    xb(:,size(xb,2)-size(block_exogenous_controls,2)+1:size(xb,2))=block_exogenous_controls(pp+1:T-hh,:)-block_exogenous_controls(pp:T-hh-1,:);
                end
            end
            jj=1;
            while jj <= p_max
                xb=[xb,endogenous_variables(pp+1-jj:T-hh-jj,:)];
                if ld == 1
                    xb(:,size(xb,2)-N+1:size(xb,2))=endogenous_variables(pp+1-jj:T-hh-jj,:)-endogenous_variables(pp-jj:T-hh-1-jj,:);
                end
                if controls ~= 0
                    xb=[xb,controls(pp+1-jj:T-hh-jj,:)];
                    if ld == 1
                        xb(:,size(xb,2)-size(controls,2)+1:size(xb,2))=controls(pp+1-jj:T-hh-jj,:)-controls(pp-jj:T-hh-1-jj,:);
                    end
                end
                if block_exogenous_controls ~= 0
                    xb=[xb,block_exogenous_controls(pp+1-jj:T-hh-jj,:)];
                    if ld == 1
                        xb(:,size(xb,2)-size(block_exogenous_controls,2)+1:size(xb,2))=block_exogenous_controls(pp+1-jj:T-hh-jj,:)-block_exogenous_controls(pp-jj:T-hh-1-jj,:);
                    end
                end
                jj=jj+1;
            end

            z_base_iv{hh+1,ii} = [shock_series(pp+1:T-hh,:), xb(:,l+1:size(xb,2))];

            xb = [xb, zeros(size(xb,1),l)];
            x_base_iv{hh+1,ii}    = xb;
            b_precomp_iv(hh+1,ii) = size(xb,2);

            z11_base_iv{hh+1,ii} = z_base_iv{hh+1,ii};
            if has_dummies1
                z11_base_iv{hh+1,ii} = [z11_base_iv{hh+1,ii}, dummies1(pp+1+hh:T,:)];
            end

            xt=[];
            if leads==1
                jj=1;
                while jj <= min(p_max,hh)
                    xt=[xt,shock_series(pp+1+jj:T-hh+jj,:)];
                    jj=jj+1;
                end
            end
            zt=[];
            if has_dummies1 && ii==1
                xt=[xt,dummies1(pp+1+hh:T,:)];
                zt=[zt,dummies1(pp+1+hh:T,:)];
            end
            if has_dummies2 && ii==2
                xt=[xt,dummies2(pp+1+hh:T,:)];
                zt=[zt,dummies2(pp+1+hh:T,:)];
            end
            x_tail_iv{hh+1,ii} = xt;
            z_tail_iv{hh+1,ii} = zt;

            if ac == 1
                if hh>0
                    acj0 = 0;
                    if hh>control_truncation
                        acj0 = hh-control_truncation;
                    end
                    ac_width = hh - acj0;
                else
                    ac_width = 0;
                end
            elseif ac == 2
                if hh>0
                    ac_width = N;
                else
                    ac_width = 0;
                end
            else
                ac_width = 0;
            end

            bb   = b_precomp_iv(hh+1,ii);
            n_bh = bb + ac_width + size(xt,2);

            b0_m_bh = zeros(n_bh,1);
            b0_v_bh = 10000000000*eye(n_bh);

            if minnesota_lr==1
                b0_v_bh(1:l,1:l) = (sige2(ii)*(lambda_irf/(1+hh))^2)*eye(l);
            end

            if minnesota_lags==1
                acj=1;
                while acj<=p_max
                    ack=1;
                    while ack<=N
                        b0_v_bh(l+1+trend+(acj-1)*N+ack,l+1+trend+(acj-1)*N+ack)=(sige2(ii)/sige2(ack))*(lambda_lags/acj)^2;
                        if acj==1 && ack==ii
                            b0_m_bh(l+1+trend+(acj-1)*N+ack)=1;
                        end
                        ack=ack+1;
                    end
                    acj=acj+1;
                end
            end

            if ac == 1
                if minnesota_controls == 1
                    if hh>0
                        acj=0; acjj=0;
                        if hh>control_truncation
                            acjj=hh-control_truncation;
                        end
                        while acjj<=hh-1
                            b0_m_bh(bb+1+acj)=mean_decay^(hh-1-acjj);
                            b0_v_bh(bb+1+acj,bb+1+acj)=(lambda_controls/(hh-acjj))^2;
                            acj=acj+1;
                            acjj=acjj+1;
                        end
                    end
                end
            end

            if ac == 2
                if minnesota_controls == 1
                    if hh>0
                        b0_m_bh(bb+1:bb+N)=zeros(N,1); b0_m_bh(bb+ii)=1;
                        b0_v_bh(bb+1:bb+N,bb+1:bb+N)=(lambda_controls^2)*eye(N);
                    end
                end
            end

            if impact_prior==1
                if hh==0
                    if ii==1 && proxy_first==1
                        b0_m_bh(1:l)=norm_sign(1:l);
                        b0_v_bh(1:l,1:l)=sige2(ii)*(lambda_impact^2)*eye(l);
                    else
                        b0_v_bh(1:l,1:l)=sige2(ii)*(lambda_impact_others^2)*eye(l);
                    end
                end
            end

            b0_v_base_iv{hh+1,ii} = b0_v_bh;
            b0_m_base_iv{hh+1,ii} = b0_m_bh;

        end
    end
end

%% Precompute for ps==1 & iv==0
%
%  ps==1 (Piger-Stockwell's alternative controls) changes the lag
%  structure of x: instead of appending all N variables' lags at each lag
%  length j (interleaved with controls/block-exog/leads), it appends only
%  the own-variable lagged DIFFERENCE at each j (and only when ld==1 --
%  faithfully reproduced as-is, including that ld==0 & ps==1 adds no
%  endogenous-lag term at all, exactly like the original), while
%  accumulating every OTHER variable's lagged LEVEL into x_other_lags
%  across the whole lag loop and appending that block once at the end.
%  None of that depends on the draw -- only on (h,i) and the fixed inputs
%  -- so it precomputes the same way x_base does for ps==0. This only
%  covers iv==0; ps==1 & iv==1 falls back to the original inline code.
if precompute_x_ps
    p_precomp_ps = p_max + (ld==1); % matches what p would be at this point below

    x_base_ps    = cell(h_max+1, N);
    x_tail_ps    = cell(h_max+1, N);
    y_base_ps    = cell(h_max+1, N);
    b_precomp_ps = zeros(h_max+1, N);

    b0_v_base_ps = cell(h_max+1, N);
    b0_m_base_ps = cell(h_max+1, N);

    for hh = 0:h_max
        for ii = 1:N

            pp = p_precomp_ps;

            if ld == 1
                y_base_ps{hh+1,ii} = endogenous_variables(pp+1+hh:T,ii)-endogenous_variables(pp+1-1:T-hh-1,ii);
            else
                y_base_ps{hh+1,ii} = endogenous_variables(pp+1+hh:T,ii);
            end

            xb=[shock_series(pp+1:T-hh,1:k),ones(T-pp-hh,1),repmat((1:1:T-pp-hh)',1,trend)];
            if block_exogenous_controls ~= 0
                xb=[xb,block_exogenous_controls(pp+1:T-hh,:)];
            end

            x_other_lags_ps = [];
            jj=1;
            while jj <= p_max
                if ld == 1
                    xb=[xb,endogenous_variables(pp+1-jj:T-hh-jj,ii)-endogenous_variables(pp-jj:T-hh-1-jj,ii)];
                end
                if ii==1
                    x_other_lags_ps = [x_other_lags_ps, endogenous_variables(pp+1-jj:T-hh-jj,ii+1:N)];
                else
                    x_other_lags_ps = [x_other_lags_ps, endogenous_variables(pp+1-jj:T-hh-jj,1:ii-1), endogenous_variables(pp+1-jj:T-hh-jj,ii+1:N)];
                end
                if controls ~= 0
                    xb=[xb,controls(pp+1-jj:T-hh-jj,:)];
                end
                if block_exogenous_controls ~= 0
                    xb=[xb,block_exogenous_controls(pp+1-jj:T-hh-jj,:)];
                end
                if leads==1
                    xb=[xb,shock_series(pp+1-jj:T-hh-jj,:)];
                end
                jj=jj+1;
            end

            xb = [xb, x_other_lags_ps];

            x_base_ps{hh+1,ii}    = xb;
            b_precomp_ps(hh+1,ii) = size(xb,2);

            xt=[];
            if leads==1
                jj=1;
                while jj <= min(p_max,hh)
                    xt=[xt,shock_series(pp+1+jj:T-hh+jj,:)];
                    jj=jj+1;
                end
            end
            if has_dummies1 && ii==1
                xt=[xt,dummies1(pp+1+hh:T,:)];
            end
            if has_dummies2 && ii==2
                xt=[xt,dummies2(pp+1+hh:T,:)];
            end
            x_tail_ps{hh+1,ii} = xt;

            if ac == 1
                if hh>0
                    acj0 = 0;
                    if hh>control_truncation
                        acj0 = hh-control_truncation;
                    end
                    ac_width = hh - acj0;
                else
                    ac_width = 0;
                end
            elseif ac == 2
                if hh>0
                    ac_width = N;
                else
                    ac_width = 0;
                end
            else
                ac_width = 0;
            end

            bb   = b_precomp_ps(hh+1,ii);
            n_bh = bb + ac_width + size(xt,2);

            b0_m_bh = zeros(n_bh,1);
            b0_v_bh = 10000000000*eye(n_bh);

            if minnesota_lr==1
                b0_v_bh(1:l,1:l) = (sige2(ii)*(lambda_irf/(1+hh))^2)*eye(l);
            end

            if minnesota_lags==1
                acj=1;
                while acj<=p_max
                    ack=1;
                    while ack<=N
                        b0_v_bh(l+1+trend+(acj-1)*N+ack,l+1+trend+(acj-1)*N+ack)=(sige2(ii)/sige2(ack))*(lambda_lags/acj)^2;
                        if acj==1 && ack==ii
                            b0_m_bh(l+1+trend+(acj-1)*N+ack)=1;
                        end
                        ack=ack+1;
                    end
                    acj=acj+1;
                end
            end

            if ac == 1
                if minnesota_controls == 1
                    if hh>0
                        acj=0; acjj=0;
                        if hh>control_truncation
                            acjj=hh-control_truncation;
                        end
                        while acjj<=hh-1
                            b0_m_bh(bb+1+acj)=mean_decay^(hh-1-acjj);
                            b0_v_bh(bb+1+acj,bb+1+acj)=(lambda_controls/(hh-acjj))^2;
                            acj=acj+1;
                            acjj=acjj+1;
                        end
                    end
                end
            end

            if ac == 2
                if minnesota_controls == 1
                    if hh>0
                        b0_m_bh(bb+1:bb+N)=zeros(N,1); b0_m_bh(bb+ii)=1;
                        b0_v_bh(bb+1:bb+N,bb+1:bb+N)=(lambda_controls^2)*eye(N);
                    end
                end
            end

            if impact_prior==1
                if hh==0
                    if ii==1 && proxy_first==1
                        b0_m_bh(1:l)=norm_sign(1:l);
                        b0_v_bh(1:l,1:l)=sige2(ii)*(lambda_impact^2)*eye(l);
                    else
                        b0_v_bh(1:l,1:l)=sige2(ii)*(lambda_impact_others^2)*eye(l);
                    end
                end
            end

            b0_v_base_ps{hh+1,ii} = b0_v_bh;
            b0_m_base_ps{hh+1,ii} = b0_m_bh;

        end
    end
end

%% Main Gibbs loop (verbatim from BLP_Gibbs_linear_NZ_slow.m, g/h/i structure)
g=1;
while g <= CAPN

    if show_progress && mod(g,print_every)==0
        %Show progress on screen even while BLP_replication_run.m has a
        %diary log open (diary mirrors everything sent to the screen to
        %that file for as long as it's on), without it ending up in
        %replication_results.txt: toggle diary off just for this fprintf,
        %then straight back on again right after (bare "diary on" resumes
        %the same log file MATLAB already has open, no path needed here).
        diary_was_on = strcmp(get(0,'Diary'),'on');
        if diary_was_on
            diary off
        end
        fprintf('  Gibbs draw %d/%d\n', g, CAPN);
        if diary_was_on
            diary on
        end
    end

    x_other_lags=[];

    irfg   = zeros(l*N*(h_max+1),1);
    corrg  = zeros(N*(h_max+1),1);
    corr5g = zeros(N*(h_max+1),1);
    corr2g = zeros(N*(h_max+1),1);
    corrhg = zeros(h_max+1,1);
    if smoothing == 1
        irfdiffg = zeros(k*N*(h_max+1),1);
    end
    if ac == 1
        errors_g = zeros(T-(ld==1), N*(h_max+1));
    end

    h=0;
    while h<=h_max

        p_for_size = p_max + (ld==1);
        errors_hg = zeros(T-p_for_size-h, N);

        i=1;
        while i<=N

            p = p_max;

            if ld == 1
                p=p_max+1;
            end

            if precompute_x
                y = y_base{h+1,i};
            elseif precompute_x_iv
                y = y_base_iv{h+1,i};
            elseif precompute_x_ps
                y = y_base_ps{h+1,i};
            else
                if ld == 1
                    y=endogenous_variables(p+1+h:T,i)-endogenous_variables(p+1-1:T-h-1,i);
                else
                    y=endogenous_variables(p+1+h:T,i);
                end
            end

            if iv == 1 && precompute_x_iv

                instrumented_variables = instrumented_variables_iv_fixed;

                x = x_base_iv{h+1,i};   % includes the trailing l zero
                                        % columns overwritten with v later
                b = b_precomp_iv(h+1,i);
                z = z_base_iv{h+1,i};
                z11 = z11_base_iv{h+1,i};

                x_hat = x;

                if ac == 1
                    if h>0
                        j=0;
                        if h>control_truncation
                            j=h-control_truncation;
                        end
                        while j<=h-1
                            x_hat=[x_hat,errors_g(p+1+j:T-h+j,N*j+i)];
                            j=j+1;
                        end
                    end
                end

                if ac == 2
                    if h>0
                        x_hat=[x_hat,errors_g(p+h:T-1,N*(h-1)+1:N*(h-1)+N)];
                    end
                end

                n_xnd=size(x_hat,2);

                x_hat = [x_hat, x_tail_iv{h+1,i}];
                z     = [z,     z_tail_iv{h+1,i}];

                if smoothing == 1
                    if h>0
                        y=y-(instrumented_variables(p+1:T-h,:)-instrumented_variables(p:T-h-1,:))*prmtr_level_store((h-1)*N+i,1:l)';
                    end
                end

            elseif iv == 1

                instrumented_variables = norm_sign(1)*endogenous_variables(:,1);
                if l>1
                    instrumented_variables = [];
                    for m=1:l
                        grp = (m-1)*group_size+1:m*group_size; %side m's own group of instrument columns within shock_series
                        instrumented_variables = [instrumented_variables,norm_sign(m)*endogenous_variables(:,1).*any(shock_series(:,grp)>0,2)]; %any(...) unions across a side's instrument columns since they need not all be nonzero at exactly the same times
                    end
                end

                x=[instrumented_variables(p+1:T-h,:)-instrumented_variables(p:T-h-1,:),ones(T-p-h,1),repmat((1:1:T-p-h)',1,trend)];
                if block_exogenous_controls ~= 0
                    x=[x,block_exogenous_controls(p+1:T-h,:)];
                    if ld ==1 && ps == 0
                        x(:,size(x,2)-size(block_exogenous_controls,2)+1:size(x,2))=block_exogenous_controls(p+1:T-h,:)-block_exogenous_controls(p:T-h-1,:);
                    end
                end
                j=1;
                while j <= p_max
                    if ps == 0
                        x=[x,endogenous_variables(p+1-j:T-h-j,:)];
                    end
                    if ld == 1
                        if ps == 1
                            x=[x,endogenous_variables(p+1-j:T-h-j,i)-endogenous_variables(p-j:T-h-1-j,i)];
                        else
                            x(:,size(x,2)-N+1:size(x,2))=endogenous_variables(p+1-j:T-h-j,:)-endogenous_variables(p-j:T-h-1-j,:);
                        end
                    end
                    if ps == 1
                        if i==1
                            x_other_lags = [x_other_lags endogenous_variables(p+1-j:T-h-j,i+1:N)];
                        else
                            x_other_lags = [x_other_lags endogenous_variables(p+1-j:T-h-j,1:i-1) endogenous_variables(p+1-j:T-h-j,i+1:N)];
                        end
                    end
                    if controls ~= 0
                        x=[x,controls(p+1-j:T-h-j,:)];
                        if ld == 1 && ps == 0
                            x(:,size(x,2)-size(controls,2)+1:size(x,2))=controls(p+1-j:T-h-j,:)-controls(p-j:T-h-1-j,:);
                        end
                    end
                    if block_exogenous_controls ~= 0
                        x=[x,block_exogenous_controls(p+1-j:T-h-j,:)];
                        if ld == 1 && ps == 0
                            x(:,size(x,2)-size(block_exogenous_controls,2)+1:size(x,2))=block_exogenous_controls(p+1-j:T-h-j,:)-block_exogenous_controls(p-j:T-h-1-j,:);
                        end
                    end
                    j=j+1;
                end

                if ps == 1
                    x=[x,x_other_lags];
                    x_other_lags=[];
                end

                x_hat=[x,zeros(length(x),l)];
                z=[shock_series(p+1:T-h,:),x(:,l+1:size(x,2))];
                b = size(x_hat,2);

                if ac == 1
                    if h>0
                        j=0;
                        if h>control_truncation
                            j=h-control_truncation;
                        end
                        while j<=h-1
                            x_hat=[x_hat,errors_g(p+1+j:T-h+j,N*j+i)];
                            j=j+1;
                        end
                    end
                end

                if ac == 2
                    if h>0
                        x_hat=[x_hat,errors_g(p+h:T-1,N*(h-1)+1:N*(h-1)+N)];
                    end
                end

                n_xnd=size(x_hat,2);

                z11=z;
                if has_dummies1
                    z11=[z11,dummies1(p+1+h:T,:)];
                end

                if leads==1
                    j=1;
                    while j <= min(p_max,h)
                        x_hat=[x_hat,shock_series(p+1+j:T-h+j,:)];
                        j=j+1;
                    end
                end

                if has_dummies1 && i==1
                    x_hat=[x_hat,dummies1(p+1+h:T,:)];
                    z=[z,dummies1(p+1+h:T,:)];
                end

                if has_dummies2 && i==2
                    x_hat=[x_hat,dummies2(p+1+h:T,:)];
                    z=[z,dummies2(p+1+h:T,:)];
                end

                if smoothing == 1
                    if h>0
                        y=y-(instrumented_variables(p+1:T-h,:)-instrumented_variables(p:T-h-1,:))*prmtr_level_store((h-1)*N+i,1:l)';
                    end
                end

            elseif precompute_x

                x = x_base{h+1,i};
                b = b_precomp(h+1,i);

                if ac == 1
                    if h>0
                        j=0;
                        if h>control_truncation
                            j=h-control_truncation;
                        end
                        while j<=h-1
                            x=[x,errors_g(p+1+j:T-h+j,N*j+i)];
                            j=j+1;
                        end
                    end
                end

                if ac == 2
                    if h>0
                        x=[x,errors_g(p+h:T-1,N*(h-1)+1:N*(h-1)+N)];
                    end
                end

                n_xnd=size(x,2);

                x = [x, x_tail{h+1,i}];

                if smoothing == 1
                    if h>0
                        y=y-shock_series(p+1:T-h,1:k)*prmtr_level_store(N*(h-1)+i,1:k)';
                    end
                end

                x_hat=x;

            elseif precompute_x_ps

                x = x_base_ps{h+1,i};
                b = b_precomp_ps(h+1,i);

                if ac == 1
                    if h>0
                        j=0;
                        if h>control_truncation
                            j=h-control_truncation;
                        end
                        while j<=h-1
                            x=[x,errors_g(p+1+j:T-h+j,N*j+i)];
                            j=j+1;
                        end
                    end
                end

                if ac == 2
                    if h>0
                        x=[x,errors_g(p+h:T-1,N*(h-1)+1:N*(h-1)+N)];
                    end
                end

                n_xnd=size(x,2);

                x = [x, x_tail_ps{h+1,i}];

                if smoothing == 1
                    if h>0
                        y=y-shock_series(p+1:T-h,1:k)*prmtr_level_store(N*(h-1)+i,1:k)';
                    end
                end

                x_hat=x;

            else

                x=[shock_series(p+1:T-h,1:k),ones(T-p-h,1),repmat((1:1:T-p-h)',1,trend)];
                if block_exogenous_controls ~= 0
                    x=[x,block_exogenous_controls(p+1:T-h,:)];
                    if ld ==1 && ps == 0
                        x(:,size(x,2)-size(block_exogenous_controls,2)+1:size(x,2))=block_exogenous_controls(p+1:T-h,:)-block_exogenous_controls(p:T-h-1,:);
                    end
                end
                j=1;
                while j <= p_max
                    if ps == 0
                        x=[x,endogenous_variables(p+1-j:T-h-j,:)];
                    end
                    if ld == 1
                        if ps == 1
                            x=[x,endogenous_variables(p+1-j:T-h-j,i)-endogenous_variables(p-j:T-h-1-j,i)];
                        else
                            x(:,size(x,2)-N+1:size(x,2))=endogenous_variables(p+1-j:T-h-j,:)-endogenous_variables(p-j:T-h-1-j,:);
                        end
                    end
                    if ps == 1
                        if i == 1
                            x_other_lags = [x_other_lags endogenous_variables(p+1-j:T-h-j,i+1:N)];
                        else
                            x_other_lags = [x_other_lags endogenous_variables(p+1-j:T-h-j,1:i-1) endogenous_variables(p+1-j:T-h-j,i+1:N)];
                        end
                    end
                    if controls ~= 0
                        x=[x,controls(p+1-j:T-h-j,:)];
                        if ld == 1 && ps == 0
                            x(:,size(x,2)-size(controls,2)+1:size(x,2))=controls(p+1-j:T-h-j,:)-controls(p-j:T-h-1-j,:);
                        end
                    end
                    if block_exogenous_controls ~= 0
                        x=[x,block_exogenous_controls(p+1-j:T-h-j,:)];
                        if ld == 1 && ps == 0
                            x(:,size(x,2)-size(block_exogenous_controls,2)+1:size(x,2))=block_exogenous_controls(p+1-j:T-h-j,:)-block_exogenous_controls(p-j:T-h-1-j,:);
                        end
                    end
                    if leads==1
                        x=[x,shock_series(p+1-j:T-h-j,:)];
                    end
                    j=j+1;
                end

                if ps == 1
                    x=[x,x_other_lags];
                    x_other_lags=[];
                end

                b=size(x,2);

                if ac == 1
                    if h>0
                        j=0;
                        if h>control_truncation
                            j=h-control_truncation;
                        end
                        while j<=h-1
                            x=[x,errors_g(p+1+j:T-h+j,N*j+i)];
                            j=j+1;
                        end
                    end
                end

                if ac == 2
                    if h>0
                        x=[x,errors_g(p+h:T-1,N*(h-1)+1:N*(h-1)+N)];
                    end
                end

                n_xnd=size(x,2);

                if leads==1
                    j=1;
                    while j <= min(p_max,h)
                        x=[x,shock_series(p+1+j:T-h+j,:)];
                        j=j+1;
                    end
                end

                if has_dummies1 && i==1
                    x=[x,dummies1(p+1+h:T,:)];
                end

                if has_dummies2 && i==2
                    x=[x,dummies2(p+1+h:T,:)];
                end

                if smoothing == 1
                    if h>0
                        y=y-shock_series(p+1:T-h,1:k)*prmtr_level_store(N*(h-1)+i,1:k)';
                    end
                end

                x_hat=x;

            end

            p=p_max;

            %=================  PRIOR PARAMETERS ====================
            if precompute_x
                % b0_v_base/b0_m_base already have minnesota_lr,
                % minnesota_lags, the ac-controls prior, and impact_prior
                % baked in (see the precompute block above) -- everything
                % that does not depend on the draw. Assigning out of the
                % cell copies the stored template (MATLAB array
                % assignment is by value), so modifying b0_v/b0_m below
                % for the smoothing case can't corrupt the stored base.
                b0_m = b0_m_base{h+1,i};
                b0_v = b0_v_base{h+1,i};

                if smoothing == 1
                    if h<peak && h>0
                        b0_v(1:l,1:l) = (sige2(i)*(lambda_smooth/(base+peak-h))^2)*eye(l);
                        b0_m(1:l) = (1-gamma_lr^(peak-h))*prmtr_level_store(N*(h-1)+i,1:l)';
                    end
                    if h>=peak && h>0
                        b0_v(1:l,1:l) = (sige2(i)*(lambda_smooth/(base+h-peak))^2)*eye(l);
                        b0_m(1:l) = (gamma_lr^(h-peak)-1)*prmtr_level_store(N*(h-1)+i,1:l)';
                    end
                end
            elseif precompute_x_iv
                b0_m = b0_m_base_iv{h+1,i};
                b0_v = b0_v_base_iv{h+1,i};

                if smoothing == 1
                    if h<peak && h>0
                        b0_v(1:l,1:l) = (sige2(i)*(lambda_smooth/(base+peak-h))^2)*eye(l);
                        b0_m(1:l) = (1-gamma_lr^(peak-h))*prmtr_level_store(N*(h-1)+i,1:l)';
                    end
                    if h>=peak && h>0
                        b0_v(1:l,1:l) = (sige2(i)*(lambda_smooth/(base+h-peak))^2)*eye(l);
                        b0_m(1:l) = (gamma_lr^(h-peak)-1)*prmtr_level_store(N*(h-1)+i,1:l)';
                    end
                end
            elseif precompute_x_ps
                b0_m = b0_m_base_ps{h+1,i};
                b0_v = b0_v_base_ps{h+1,i};

                if smoothing == 1
                    if h<peak && h>0
                        b0_v(1:l,1:l) = (sige2(i)*(lambda_smooth/(base+peak-h))^2)*eye(l);
                        b0_m(1:l) = (1-gamma_lr^(peak-h))*prmtr_level_store(N*(h-1)+i,1:l)';
                    end
                    if h>=peak && h>0
                        b0_v(1:l,1:l) = (sige2(i)*(lambda_smooth/(base+h-peak))^2)*eye(l);
                        b0_m(1:l) = (gamma_lr^(h-peak)-1)*prmtr_level_store(N*(h-1)+i,1:l)';
                    end
                end
            else
                b0_m = zeros(size(x_hat,2),1);
                b0_v = 10000000000*eye(size(x_hat,2));

                if minnesota_lr==1
                    b0_v(1:l,1:l) = (sige2(i)*(lambda_irf/(1+h))^2)*eye(l);
                end

                if smoothing == 1
                    if h<peak && h>0
                        b0_v(1:l,1:l) = (sige2(i)*(lambda_smooth/(base+peak-h))^2)*eye(l);
                        b0_m(1:l) = (1-gamma_lr^(peak-h))*prmtr_level_store(N*(h-1)+i,1:l)';
                    end
                    if h>=peak && h>0
                        b0_v(1:l,1:l) = (sige2(i)*(lambda_smooth/(base+h-peak))^2)*eye(l);
                        b0_m(1:l) = (gamma_lr^(h-peak)-1)*prmtr_level_store(N*(h-1)+i,1:l)';
                    end
                end

                if minnesota_lags==1
                    j=1;
                    while j<=p
                        ii=1;
                        while ii<=N
                            b0_v(l+1+trend+(j-1)*N+ii,l+1+trend+(j-1)*N+ii)=(sige2(i)/sige2(ii))*(lambda_lags/j)^2;
                            if j==1 && ii==i
                                b0_m(l+1+trend+(j-1)*N+ii)=1;
                            end
                            ii=ii+1;
                        end
                        j=j+1;
                    end
                end

                if ac == 1
                    if minnesota_controls == 1
                        if h>0
                            j=0; jj=0;
                            if h>control_truncation
                                jj=h-control_truncation;
                            end
                            while jj<=h-1
                                b0_m(b+1+j)=mean_decay^(h-1-jj);
                                b0_v(b+1+j,b+1+j)=(lambda_controls/(h-jj))^2;
                                j=j+1;
                                jj=jj+1;
                            end
                        end
                    end
                end

                if ac == 2
                    if minnesota_controls == 1
                        if h>0
                            b0_m(b+1:b+N)=zeros(N,1); b0_m(b+i)=1;
                            b0_v(b+1:b+N,b+1:b+N)=(lambda_controls^2)*eye(N);
                        end
                    end
                end

                if impact_prior==1
                    if h==0
                        if i==1 && proxy_first==1
                            b0_m(1:l)=norm_sign(1:l);
                            b0_v(1:l,1:l)=sige2(i)*(lambda_impact^2)*eye(l);
                        else
                            b0_v(1:l,1:l)=sige2(i)*(lambda_impact_others^2)*eye(l);
                        end
                    end
                end
            end

            if iv == 1
                b0s1_m = zeros(size(z,2),1);
                b0s1_v = 10000000000*eye(size(z,2));
                if prior_iv==1
                    b0s1_m(1:k)=iv_prior_mean*norm_sign_z; %norm_sign_z (length k, one entry per instrument column), not norm_sign (length l, one per side) -- see norm_sign_z's definition above
                    b0s1_v(1:k,1:k)=iv_prior_var*eye(k);
                end
            end

            v0 = 0; d0 = 0;
            v0_iv=v0; d0_iv=d0;

            if minnesota_variance == 1
                v0 = lambda_var; v0_iv=v0;
                if horizon_expand == 1
                    d0 = (v0+2)*sige2(i)*(h+1);
                    d0_iv=(v0_iv+2)*sige2(1)*(h+1);
                else
                    d0 = (v0+2)*sige2(i);
                    d0_iv=(v0_iv+2)*sige2(1);
                end
            end

            if g==1
                betagl = prmtr_point_store((i-1)*(h_max+1)+h+1,:)';
                if iv ==1
                    betagl1 = prmtr1_point_store(1:l,:)';
                    betagl=[betagl;zeros(l,1)];
                end
                if ac == 1
                    if h>0
                        betagl=[betagl;zeros(min(h,control_truncation),1)];
                    end
                end
                if ac == 2
                    if h>0
                        betagl=[betagl;zeros(N,1)];
                    end
                end
                if leads==1 && h>0
                    betagl=[betagl;zeros(k*min(h,p),1)];
                end
                if has_dummies1 && i==1
                    betagl=[betagl;zeros(size(dummies1,2),1)];
                end
                if has_dummies2 && i==2
                    betagl=[betagl;zeros(size(dummies2,2),1)];
                end
            else
                if iv ==1
                    betagl1 = prmtr1_lag_store(1:l,:)';
                    betagl = prmtr_lag_store(N*h+i,1:n_xnd)';
                else
                    betagl = prmtr_lag_store(N*h+i,1:n_xnd)';
                end
                if leads==1 && h>0
                    betagl=[betagl;leads_lag_store(N*h*k+i,1:k*min(p,h))'];
                end
                if has_dummies1 && i==1
                    betagl=[betagl;dummies1_lag_store(h+1,:)'];
                end
                if has_dummies2 && i==2
                    betagl=[betagl;dummies2_lag_store(h+1,:)'];
                end
            end

            %sign restriction for this equation (see opts.sign_restrictions): [] keeps GEN_BETA's legacy hard-coded NZ restrictions; first-stage (IV) draws are never sign-restricted
            if isempty(sign_restrictions)
                sr_i = []; sr_first = [];
            else
                sr_i = sign_restrictions(i); sr_first = 0;
            end
            %=================START SAMPLING================
            if iv ==1

                if i==1 && h==0
                    v=[];
                    if l==1
                        sig2g1 = GEN_SIG(x(:,1),z,betagl1,v0_iv,d0_iv,prior_only);
                        betag1 = GEN_BETA(x(:,1),z,sig2g1,b0s1_m,b0s1_v,i,h,l,n_x,norm_sign,sign_lag,prior_only,[],sr_first);
                        v=x(:,1)-z*betag1(:,1);
                    else
                        betag1=zeros(size(z,2),l); %sized by size(z,2) (the instrument/regressor count each side's first-stage regression draws from), not size(x,2) (the number of instrumented variables)
                        for j=1:l
                            grp = (j-1)*group_size+1:j*group_size; %side j's own group of instrument columns (rows of betagl1/b0s1_m/b0s1_v, columns of z) -- group_size==1 reduces this exactly to the original single-column-j design
                            betagl1j=[betagl1(grp,j);betagl1(k+1:end,j)]; %k+1 (not l+1): z's shared/control block starts after all k instrument columns, which only coincided with l+1 when k==l
                            b0s1_mj=[b0s1_m(grp);b0s1_m(k+1:end)];
                            b0s1_vj=[diag(b0s1_v(grp,grp));diag(b0s1_v(k+1:end,k+1:end))].*eye(length(b0s1_mj));

                            xj=x(:,j);
                            zj=[z(:,grp),z(:,k+1:end)]; %this side's own instrument group plus the shared/control columns -- excludes the OTHER sides' instrument groups, same exclusion-restriction logic as the original single-column design
                            zj=zj(xj~=0,:);
                            xj=xj(xj~=0);

                            sig2g1 = GEN_SIG(xj,zj,betagl1j,v0_iv,d0_iv,prior_only);
                            betag1j = GEN_BETA(xj,zj,sig2g1,b0s1_mj,b0s1_vj,i,h,l,n_x,norm_sign,sign_lag,prior_only,[],sr_first);
                            betag1(grp,j) = betag1j(1:group_size);
                            betag1(k+1:end,j) = betag1j(group_size+1:end);
                            v=[v,x(:,j)-z*betag1(:,j)];

                            if j>1 && orthogonalize==1
                                vjb=(v(:,1:j-1)'*v(:,1:j-1)\v(:,1:j-1)'*v(:,j));
                            end
                        end
                    end
                end

                v=[];
                if l==1
                    v=x(:,1)-z11*betag1(:,1);
                else
                    for j=1:l
                        v=[v,x(:,j)-z11*betag1(:,j)];
                        if j>1 && orthogonalize==1
                            v(:,j)=v(:,j)-v(:,1:j-1)*vjb;
                        end
                    end
                end

                x_hat(:,b-l+1:b)=v;
                if i==1 && h==0
                    betag= [norm_sign(1:l);zeros(size(x_hat,2)-size(instrumented_variables,2),1)];
                else
                    sig2g = GEN_SIG(y,x_hat,betagl,v0,d0,prior_only);
                    betag = GEN_BETA(y,x_hat,sig2g,b0_m,b0_v,i,h,l,n_x,norm_sign,sign_lag,prior_only,[],sr_i);
                end

            else
                sig2g = GEN_SIG(y,x,betagl,v0,d0,prior_only);
                betag = GEN_BETA(y,x,sig2g,b0_m,b0_v,i,h,l,n_x,norm_sign,sign_lag,prior_only,[],sr_i);
            end

            if iv == 1
                x = x_hat;
            end
            errors=y-x*betag;
            hi_idx = N*h+i;
            errors_hg(:,i) = errors;
            corr_pair = serial_correlation_pair_fcn(errors,[1 2 5],0);
            corr_out  = corr_pair(1);
            corr2_out = corr_pair(2);
            corr5_out = corr_pair(3);
            corrg(hi_idx)  = corr_out;
            corr2g(hi_idx) = corr2_out;
            corr5g(hi_idx) = corr5_out;

            if ac == 1
                errors_g(:,N*h+i) = [zeros(p+h,1);(y-x*betag)];
            end

            if ac == 2
                if h>0
                    errors_g(:,N*(h-1)+i)=[zeros(p+h,1);(y-x*betag)];
                end
            end

            if has_dummies2 && i==2
                dummies2_lag_store(h+1,:)=betag(length(betag)-size(dummies2,2)+1:length(betag))';
                betag=betag(1:length(betag)-size(dummies2,2));
            end

            if has_dummies1 && i==1
                dummies1_lag_store(h+1,:)=betag(length(betag)-size(dummies1,2)+1:length(betag))';
                betag=betag(1:length(betag)-size(dummies1,2));
            end

            if leads==1 && h>0
                leads_lag_store(N*h*k+i,1:k*min(p,h))=betag(length(betag)-k*min(p,h)+1:length(betag))';
                betag=betag(1:length(betag)-k*min(p,h));
            end

            prmtr_lag_store(N*h+i,1:length(betag))=betag';
            if iv == 1
                prmtr1_lag_store(1:l,1:length(betag1))=betag1';
            end

            if smoothing == 1
                irfdiffg((hi_idx-1)*k+1:hi_idx*k) = betag(1:k);
                if h>0
                    betag(1:l)=prmtr_level_store(N*(h-1)+i,1:l)'+betag(1:l);
                end
                prmtr_level_store(N*h+i,1:length(betag))=betag';
            end

            irfg((hi_idx-1)*l+1:hi_idx*l) = betag(1:l);

            i=i+1;
        end

        corr_i_out=(sum(sum(abs(corr(errors_hg))))-N)/(N^2-N+eps);
        corrhg(h+1) = corr_i_out;

        h=h+1;
    end

    if g>N0
        row = g - N0;
        irfmm(row,:) = irfg';
        if smoothing == 1
            irfdiffmm(row,:) = irfdiffg';
        end
        corrmm(row,:)  = corrg';
        corr5mm(row,:) = corr5g';
        corr2mm(row,:) = corr2g';
        corrhmm(row,:) = corrhg';
    end

    g=g+1;
end

%% Posterior moments (equal-tailed bands, same as original script)
fnl_mat = irfmm;
posterior_moments = moments_fcn(fnl_mat,c_level);

Bayes_irfs=[]; Bayes_irfs_median=[]; Bayes_irfs_lower=[]; Bayes_irfs_upper=[];
for h=0:h_max
    Bayes_irfs        = [Bayes_irfs,        posterior_moments(l*N*h+1:l*N*h+l*N,1)];
    Bayes_irfs_median = [Bayes_irfs_median, posterior_moments(l*N*h+1:l*N*h+l*N,3)];
    Bayes_irfs_lower  = [Bayes_irfs_lower,  posterior_moments(l*N*h+1:l*N*h+l*N,4)];
    Bayes_irfs_upper  = [Bayes_irfs_upper,  posterior_moments(l*N*h+1:l*N*h+l*N,5)];
end

out.irfmm             = irfmm;
if smoothing == 1
    out.irfdiffmm = irfdiffmm;
else
    out.irfdiffmm = [];
end
out.Bayes_irfs         = Bayes_irfs';
out.Bayes_irfs_median  = Bayes_irfs_median';
out.Bayes_irfs_lower   = Bayes_irfs_lower';
out.Bayes_irfs_upper   = Bayes_irfs_upper';

% Serial-correlation diagnostics (1st- and 5th-order correlation of
% residuals across draws, and cross-equation correlation at each
% horizon), computed during the loop above -- surfaced here so a wrapper
% can reproduce the serial-correlation diagnostic plots.
out.corrmm  = corrmm;
out.corr5mm = corr5mm;
out.corr2mm = corr2mm;
out.corrhmm = corrhmm;

end
