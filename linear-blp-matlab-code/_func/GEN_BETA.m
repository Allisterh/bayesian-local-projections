function beta_draw = GEN_BETA(y,x,sig2g,b0_m,b0_v,i,h,k,n_x,norm_sign,sign_lag,prior_only,fus,sign_restr)
% James Morley
% Updated November 2026 (general sign restrictions via sign_restr)
% Draws regression coefficients from conditional posterior, or from the
% prior itself when prior_only is true (see NOTE ON PRIOR_ONLY below)
%
%% INPUTS
%
%y        LHS variable
%x        RHS variables
%sig2g    Previous draw of residual variance
%v0       prior mean
%d0       prior variance
%prior_only  (optional, default false) simulate from the prior N(b0_m,b0_v)
%            instead of the data-conditioned posterior -- see note below
%fus      (optional, default [] = none) n_x-by-1 vector of extra prior
%         precision on the regime DIFFERENCE beta(j)-beta(n_x+j) of each
%         regime-split regressor j (0 = no shrinkage for that j). Adds the
%         penalty fus(j)*(beta(j)-beta(n_x+j))^2/2 to the prior on top of
%         the independent N(b0_m,b0_v) prior, so the prior precision is no
%         longer diagonal for those coefficients. Used only by the
%         regime-switching core (opts.lambda_regime_diff); the linear code
%         never passes it, so its behavior is unchanged.
%sign_restr (optional, default [] = legacy NZ restrictions) scalar sign
%         restriction for THIS equation's response to the shock(s) at
%         horizons h<=sign_lag: +1 = non-negative, -1 = non-positive,
%         0 = unrestricted. Stated for a shock with norm_sign=+1 and
%         flipped automatically for a shock with norm_sign=-1. When empty
%         or omitted, the original hard-coded restrictions apply (variables
%         2 and 3 non-positive, variable 4 non-negative, variable 1
%         unrestricted), so existing callers behave exactly as before.
%
%
%% OUTPUTS
%beta_draw     Draw of regression coefficients from multivariate normal
%
%% NOTE ON THE FALLBACK FOR ILL-CONDITIONED DRAWS
% A is symmetric positive definite in exact arithmetic (xstar'*xstar is
% positive semi-definite, and the prior contributes a strictly positive
% diagonal), but chol() can still throw "Matrix must be positive
% definite" on a matrix that is only *numerically* borderline -- e.g.
% when a regime split leaves one regime with very few/collinear
% observations, xstar'*xstar becomes severely ill-conditioned and
% rounding error can push its smallest computed eigenvalue just below
% zero. chol() fails hard on that; MATLAB's inv() (LU-based, no
% positive-definiteness check) does not.

    persistent warned_fallback

    if nargin<12 || isempty(prior_only)
        prior_only = false;
    end

    n = size(x,2);
    b0_v_diag_inv = 1./diag(b0_v);

    use_fusion = nargin>=13 && ~isempty(fus) && any(fus(:)~=0);
    if use_fusion
        %Full prior precision: the diagonal one plus the regime-difference
        %penalty for each j with fus(j)>0 (see the fus input above)
        P0 = diag(b0_v_diag_inv);
        jj = find(fus(:)~=0);
        f  = fus(jj); f = f(:);
        P0(sub2ind([n n], jj, jj))         = P0(sub2ind([n n], jj, jj)) + f;
        P0(sub2ind([n n], n_x+jj, n_x+jj)) = P0(sub2ind([n n], n_x+jj, n_x+jj)) + f;
        P0(sub2ind([n n], jj, n_x+jj))     = P0(sub2ind([n n], jj, n_x+jj)) - f;
        P0(sub2ind([n n], n_x+jj, jj))     = P0(sub2ind([n n], n_x+jj, jj)) - f;
    end
    draw_mode = 'precision';   % 'precision' -> R is chol(A); 'covariance' -> c is chol(b1_v) (fallback, see note above)

    if prior_only
        % Simulate from the prior: skip the data (y,x,sig2g go unused
        % below) and just use the prior mean/precision directly. b0_v is
        % assumed diagonal (as elsewhere in this file), so its Cholesky
        % factor is just the elementwise sqrt -- no need to form/factor an
        % n-by-n matrix for something this simple.
        if use_fusion
            R = chol(P0);
        else
            R = diag(sqrt(b0_v_diag_inv));
        end
        b1_m = b0_m;
    else
        ystar=y./sqrt(sig2g);
        xstar=x./sqrt(sig2g);

        A = xstar'*xstar;
        if use_fusion
            A   = A + P0;                                 % posterior precision
            rhs = P0*b0_m + xstar'*ystar;
        else
            A(1:n+1:end) = A(1:n+1:end) + b0_v_diag_inv';   % posterior precision
            rhs = b0_v_diag_inv.*b0_m + xstar'*ystar;
        end
        try
            R   = chol(A);                   % A = R'*R, R upper triangular
            b1_m = R\(R'\rhs);
        catch ME
            % Only fall back for the specific failure this is meant to
            % handle (A not numerically positive definite); anything else
            % is a real bug and should keep erroring out as before.
            if isempty(regexpi(ME.message,'positive definite','once'))
                rethrow(ME);
            end
            % A was not numerically positive definite (near-degenerate
            % regime split or similar ill-conditioning) -- fall back to
            % the old, more forgiving inv(A)/chol(b1_v) approach instead
            % of erroring out. See "NOTE ON THE FALLBACK" above.
            if isempty(warned_fallback) || ~warned_fallback
                warning('GEN_BETA:precisionFallback', ['Posterior precision matrix was not numerically positive definite ' ...
                    '(likely a near-degenerate regime split); falling back to the slower inverse-based sampling used by the ' ...
                    'pre-Sept-2026 code for this and any similarly ill-conditioned draws.']);
                warned_fallback = true;
            end
            draw_mode = 'covariance';
            b1_v = inv(A);
            b1_m = b1_v*rhs;
            c = chol(b1_v);
        end
    end

    accept=0;

    while accept==0

    if strcmp(draw_mode,'precision')
        beta_draw=b1_m + R\randn(n,1);
    else
        beta_draw=b1_m + c'*randn(n,1);
    end

accept=1;
if nargin>=14 && ~isempty(sign_restr)
    % General sign restrictions (see sign_restr above)
    if sign_restr~=0 && h<=sign_lag
        for j=1:k
            if sign_restr*norm_sign(j)*beta_draw(j)<0
                accept=0;
            end
            if n_x>0 && sign_restr*norm_sign(j)*beta_draw(n_x+j)<0   % also restrict the second regime's response when regime switching
                accept=0;
            end
        end
    end
elseif i>1 && h<=sign_lag   % legacy hard-coded NZ restrictions (unchanged)
    for j=1:k
        if n_x>0    % for varying fiscal regimes with/without sign-asymmetry
            if (i==2 || i==3) && (norm_sign(j)*beta_draw(j)>0 || norm_sign(j)*beta_draw(n_x+j)>0)  %sign restriction that GDP and inflation fall in first sign_lag quarters
            accept=0;
            end
            
            if (i==4) && (norm_sign(j)*beta_draw(j)<0 || norm_sign(j)*beta_draw(n_x+j)<0)  %sign restrictions on the first sign_lag quarters to remove exchange rate puzzle by construction
            accept=0;
            end
       
        end

        if n_x==0 % for the linear and sign-asymmetry cases    
        
            if (i==2 || i==3) && norm_sign(j)*beta_draw(j)>0  %sign restriction that GDP and inflation falls
            accept=0;
            end

            if (i==4) && (norm_sign(j)*beta_draw(j)<0) %sign restriction to remove exchange rate puzzle
            accept=0;
            end
        end
    end    
end

    end