df = readtable('pima-indians-diabetes.csv');
X_train = df(:,1:end-1);
X_train = table2array(X_train);

y_train = df(:,end);
y_train = table2array(y_train);

x = X_train';
t = y_train';

training_function = {'trainlm';'trainscg'};
activation_function = {'poslin';'tansig'};
regularization = [0;.25;.50];
net.trainParam.epochs = 50;

network_structure = {[4096,4096],
    [4096,2048,1024,528,256], [4096,2048,1024],
    [528,528,528,528,528],[528,528,528],
    [256,528,1024,528,256],[528,1024,528],
    [256,528,1024,528,256],[528,1024,528],
    }

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainlm';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network
net = patternnet([5,5,5,5], trainFcn);
n_layers = size(net.layers,1)-1;
net.layers{1:n_layers}.transferFcn = 'poslin';
net.performParam.regularization = 0.5;

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.input.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivision
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'crossentropy';  % Cross-Entropy

% Train the Network
[net,tr] = train(net,x,t);

% Accuracy
predict = round(net(X_train'));
accuracy = sum(y_train == round(predict'))/length(y_train);

disp(['Accuracy: ', num2str(accuracy)])


% % Choose Plot Functions
% % For a list of all plot functions type: help nnplot
% net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
%     'plotconfusion', 'plotroc'};
