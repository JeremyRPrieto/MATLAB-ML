function [Dist] = CentroidMethodGen(B_Est)

n = length(B_Est(1,:));
numer = 0;

for i = 1:n

   numer = numer + (B_Est(:,i) - 1/n) .^ 2;

end

Dist = sqrt( numer ) / sqrt( 1 - 1/n );

end

