function var_draw = GEN_SIG(y,x,betagl,v0,d0,prior_only)
% James Morley
% Updated 29 September 2026
% Draws residual variance from conditional posterior for linear
% regression, or from the prior IG(v0/2,d0/2) itself when prior_only is
% true
%
%% INPUTS
%
%y        LHS variable
%x        RHS variables
%betagl   Previous draw of regression coefficients
%v0       prior hyperparameter
%d0       prior hyperparameter
%prior_only  (optional, default false) draw from the prior instead of
%            updating v0,d0 with the residuals -- v0,d0 must therefore be
%            a proper (non-degenerate) prior, i.e. not the v0=0,d0=0 flat
%            default used when minnesota_variance==0
%
%
%% OUTPUTS
%var_draw        Draw of residual variance from inverse gamma

if nargin<6 || isempty(prior_only)
    prior_only = false;
end

if prior_only
    v1 = v0;
    d1 = d0;
else
    e_mat=y-x*betagl;
    v1 = v0 + length(e_mat);
    d1 = d0 + e_mat'*e_mat;
end
c = rndc(v1);
t2 = c/d1;
var_draw=1/t2;
