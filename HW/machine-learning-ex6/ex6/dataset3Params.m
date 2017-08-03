function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
terror = 1e20;
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% (ones(1,8)*0.03).*(3.^(0:7))

for cc=0:5
    for ss=1:5        
        c=0.03*3^cc;
        s=0.05*2^ss;
        model = svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, s));
        pred  = svmPredict(model, Xval);
        error = mean(double(pred ~= yval));
        
        if terror>error
            terror=error;
            C=c;
            sigma=s;
        end
    end
end

% =========================================================================

end
