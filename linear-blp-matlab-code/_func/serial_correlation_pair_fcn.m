function corr_out = serial_correlation_pair_fcn(tmpm1,orders,mn_out)
% James Morley
% Added Sept 2026; vectorized Sept 2026
% Same computation as serial_correlation_fcn, but returns the serial
% correlation for several lag orders from a single call instead of one
% call per order.
%
%% INPUTS
%
%tmpm1       data to consider serial correlation in
%orders      vector of lag orders to consider (e.g. [1 5])
%mn_out      mean of data
%
%% OUTPUTS
%corr_out    vector of nth-order serial correlations, one per entry of
%            orders, in the same order as orders
%
%% NOTE
% Replaces two separate serial_correlation_fcn calls (order 1 and order
% 5) per (h,i,draw) with one call computing both, cutting the call count
% at this very hot diagnostic-only site in half. serial_correlation_fcn
% itself is left as-is; it's still used by moments_fcn.m at low call
% counts. The loop below is vectorized (elementwise products + sum()
% rather than an explicit accumulation loop) for the same reason -- same
% terms, same values, but sum()'s pairwise summation can differ from a
% strict left-to-right loop at the ~1e-15 relative level. This output
% only ever feeds the serial-correlation diagnostic plots, never the
% sampler, so that's immaterial here.

nnn = length(tmpm1);
n_orders = length(orders);
corr_out = zeros(1,n_orders);

dev = tmpm1 - mn_out; % compute deviations from the mean once, reuse across orders

for oi = 1:n_orders
    order = orders(oi);
    idx_now  = (order+1):nnn;
    idx_lag  = idx_now - order;

    cov_out = sum(dev(idx_now).*dev(idx_lag));
    var_out = sum(dev(idx_now).*dev(idx_now));

    corr_out(oi) = cov_out./var_out;
end
