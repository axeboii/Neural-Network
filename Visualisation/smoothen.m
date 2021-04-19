function smoothLoss = smoothen(loss, n, m)
    numInAverages = n; 
    numOfAverages = m;
    smoothLoss = zeros(numOfAverages,1);
    for i = 1:numOfAverages
        smoothLoss(i) = mean(loss((i-1)*numInAverages + 1: i*numInAverages));
    end
end