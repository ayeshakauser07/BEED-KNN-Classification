%% ===========================================================
% KNN Classification with ROC Curve
% ============================================================

clear;
clc;
close all;

%% Load Dataset
data = readtable('BEED_data.csv');

disp('First 5 Rows:')
disp(head(data))

disp('Dataset Summary:')
summary(data)

%% Extract Features and Labels

X = table2array(data(:,1:end-1));
disp(X);
Y = data{:,end};

%% Convert Labels (if needed)

if iscell(Y) || isstring(Y)
    Y = categorical(Y);
end
%%  Standardization
mean_X = mean(X, 1);
std_X = std(X, 1, 1);

%% 
standardized_X = (X - mean_X) ./ std_X;

%% 
disp(standardized_X)
figure;

subplot(1,2,1)
bar(X(1,:))
title('Original Sample')
xlabel('Features')
ylabel('Value')
grid on

subplot(1,2,2)
bar(standardized_X(1,:))
title('Standardized Sample')
xlabel('Features')
ylabel('Z-score')
grid on
%% Plot Before and After Standardization

figure;

% Original Data
subplot(1,2,1)
bar(mean(X,1))
title('Original Data','FontSize',14,'FontWeight','bold')
xlabel('Features')
ylabel('Mean Value')
grid on

% Standardized Data
subplot(1,2,2)
bar(mean(standardized_X,1))
title('Standardized Data','FontSize',14,'FontWeight','bold')
xlabel('Features')
ylabel('Mean (Z-score)')
grid on
%% 
std_after = std(standardized_X,1,1);
disp(std_after)
%% Normalize Features

X = normalize(X);
disp(X);

%% Train-Test Split (80-20)

cv = cvpartition(Y,'HoldOut',0.20);

XTrain = X(training(cv),:);
YTrain = Y(training(cv));

XTest = X(test(cv),:);
YTest = Y(test(cv));

%% Train KNN Model

k = 5;

knnModel = fitcknn(...
    XTrain,...
    YTrain,...
    'NumNeighbors',k,...
    'Distance','euclidean',...
    'Standardize',true);

%% Predict Labels and Scores

[YPred,scores] = predict(knnModel,XTest);

%% Accuracy

accuracy = mean(YPred==YTest)*100;

fprintf('\n====================================\n');
fprintf('Accuracy = %.2f %%\n',accuracy);
fprintf('====================================\n');

%% Confusion Matrix

figure;
confusionchart(YTest,YPred);
title('Confusion Matrix');

confMat = confusionmat(YTest,YPred);

disp('Confusion Matrix:')
disp(confMat)

%% Binary Metrics

if size(confMat,1)==2

    TN = confMat(1,1);
    FP = confMat(1,2);
    FN = confMat(2,1);
    TP = confMat(2,2);

    Precision = TP/(TP+FP);
    Recall = TP/(TP+FN);
    Specificity = TN/(TN+FP);
    F1 = 2*(Precision*Recall)/(Precision+Recall);

    fprintf('\nPrecision   : %.4f\n',Precision);
    fprintf('Recall      : %.4f\n',Recall);
    fprintf('Specificity : %.4f\n',Specificity);
    fprintf('F1 Score    : %.4f\n',F1);

end

%% ===========================================================
% ROC Curve
% ============================================================

classes = unique(YTest);
numClasses = numel(classes);

figure;
hold on;

AUC = zeros(numClasses,1);

for i = 1:numClasses

    [Xroc,Yroc,~,AUC(i)] = perfcurve(...
        YTest,...
        scores(:,i),...
        classes(i));

    plot(Xroc,...
         Yroc,...
         'LineWidth',2,...
         'DisplayName',...
         sprintf('%s (AUC = %.3f)',string(classes(i)),AUC(i)));

end

plot([0 1],[0 1],'k--','LineWidth',1);

xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve');

legend('Location','best');

grid on;
hold off;

%% Display AUC

disp('-------------------------------------')
disp('Area Under Curve (AUC)')
disp('-------------------------------------')

for i = 1:numClasses
    fprintf('%s : %.4f\n',string(classes(i)),AUC(i));
end

%% ===========================================================
% Optional: 5-Fold Cross Validation Accuracy
% ============================================================

CVModel = crossval(knnModel,'KFold',5);

cvLoss = kfoldLoss(CVModel);

cvAccuracy = (1-cvLoss)*100;

fprintf('\n5-Fold Cross Validation Accuracy = %.2f %%\n',cvAccuracy);

%% ===========================================================
% End
% ===========================================================