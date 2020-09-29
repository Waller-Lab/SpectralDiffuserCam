
function eig_b = power_iteration(A, sample_vect, num_iters)
bk = rand(size(sample_vect));
for i=1:num_iters
    bk1 = A(bk);
    bk1_norm = norm(bk1);
    
    bk = bk1./bk1_norm;
end

Mx = A(bk);
xx = transpose(bk(:))*bk(:);
eig_b = (transpose(bk(:))*Mx(:))/xx;
end