function out = soft_thresh(x,tau)

out = max(abs(x)-tau,0);
out = out.*sign(x);