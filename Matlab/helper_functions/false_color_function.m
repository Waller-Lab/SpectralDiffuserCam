function false_color = false_color_function(x)

load('false_color_calib.mat');

scaling = [1,1,2.5];
false_color = zeros(size(x, 1), size(x, 2), 3);
for i=1:64
   false_color(:,:,1) =  false_color(:,:,1) + red(i)*scaling(1)*x(:,:,i);
   false_color(:,:,2) =  false_color(:,:,2) + green(i)*scaling(2)*x(:,:,i);
   false_color(:,:,3) =  false_color(:,:,3) + blue(i)*scaling(3)*x(:,:,i);
end

false_color = false_color/max(max(max(false_color)));
return