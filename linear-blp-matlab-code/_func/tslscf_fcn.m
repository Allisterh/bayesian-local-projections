function [b,b1,u,sigu2,se,x_hat] = tslscf_fcn(y,x,z,i,h,b1,z11,l,orthogonalize,k)
% James Morley
% Updated 29 September 2026
% Calculates 2SLS estimates using control function, residuals, and standard errors
%
%% INPUTS
%
%y        LHS variable
%x        RHS variables
%z        Instruments
%k        number of raw instrument columns in z's first block (z(:,1:k)) --
%         needed (rather than just l) so that a side's own instrument
%         group and the shared/control block (z(:,k+1:end)) can be told
%         apart when a side has more than one dedicated instrument column
%         (group_size = k/l > 1); with one instrument per side (k==l,
%         the original design), this reduces exactly to the prior
%         hardcoded l==2, one-column-each behavior.
%
%% OUTPUTS
%b        IV point estimates
%u        residuals based on IV estimates
%sigu2    residual variance
%se       IV SEs
%x_hat    fitted x


if i==1 && h==0

bs1 = (z'*z)\(z'*x);

if l>1
group_size = k/l;
bs1 = zeros(size(z,2),l);
for j=1:l
    grp = (j-1)*group_size+1:j*group_size; %side j's own group of instrument columns
    xj = x(:,j);
    zj = [z(:,grp),z(:,k+1:end)]; %this side's own instruments plus the shared/control columns, excluding the other sides' instrument groups
    bsj = (zj(xj~=0,:)'*zj(xj~=0,:))\(zj(xj~=0,:)'*xj(xj~=0));
    bs1j = zeros(size(z,2),1);
    bs1j(grp) = bsj(1:group_size);
    bs1j(k+1:end) = bsj(group_size+1:end);
    bs1(:,j) = bs1j;
end
end

b1=bs1;
z11=z;

else

bs1=b1;

end

if l==1
    v=x(:,1)-z11*bs1(:,1);
else
    v=x(:,1)-z11*bs1(:,1);
    for j=2:l
        vj=x(:,j)-z11*bs1(:,j);
        vjb=(v(:,1:j-1)'*v(:,1:j-1))\(v(:,1:j-1)'*vj);
        vj_orth=vj;
        if orthogonalize==1
        vj_orth=vj-v(:,1:j-1)*vjb;
        end
        v=[v,vj_orth];
    end
end

x_hat=[x,v];
b = (x_hat'*x_hat)\(x_hat'*y);   
u = y-x_hat*b;
sigu2 = (u'*u)/(length(u)-length(b));   
se = sqrt(diag(sigu2*inv(x_hat'*x_hat)));
