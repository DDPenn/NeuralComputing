% %% Download the CIFAR-10 dataset
% Requires saveCIFAR10AsFolderOfImages.m script in same working directory
%
% if ~exist('cifar-10-batches-mat','dir')
%     cifar10Dataset = 'cifar-10-matlab';
%     disp('Downloading 174MB CIFAR-10 dataset...');   
%     websave([cifar10Dataset,'.tar.gz'],...
%         ['https://www.cs.toronto.edu/~kriz/',cifar10Dataset,'.tar.gz']);
%     gunzip([cifar10Dataset,'.tar.gz'])
%     delete([cifar10Dataset,'.tar.gz'])
%     untar([cifar10Dataset,'.tar'])
%     delete([cifar10Dataset,'.tar'])
% end
% 
% % Prepare the CIFAR-10 dataset
% if ~exist('cifar10Train','dir')
%     disp('Saving the Images in folders. This might take some time...');    
%     saveCIFAR10AsFolderOfImages('cifar-10-batches-mat', pwd, true);
% end

%% Load data into imagedatastore object

categories = {'airplane', 'automobile', 'bird', 'cat',...
                'deer','dog','frog','horse','ship','truck'};

% Load & store train data
rootFolder = 'cifar10Train'; 
imds_train = imageDatastore(fullfile(rootFolder, categories), ...
    'LabelSource', 'foldernames');            
      
% Load & store test data
rootFolder = 'cifar10Test';
imds_test = imageDatastore(fullfile(rootFolder, categories), ...
    'LabelSource', 'foldernames');

% Shuffling and Splitted/Select a subset of training data
% splitEachLabel(imgDataStore, number of img for each category,
% 'randomized')

%here we should have a train suybset of 200*10 cat =2k total img and 50*10
%for test
imds_rand_Trainsubset = splitEachLabel(imds_train,200,'randomized');
imds_rand_Testsubset = splitEachLabel(imds_test,50,'randomized');

imds_rand_Trainsubset = shuffle(imds_rand_Trainsubset);
imds_rand_Testsubset = shuffle(imds_rand_Testsubset);

cv_train_split = 0.70;

% Confirm partition worked and check for class imbalance
%countEachLabel(imds_rand_Trainsubset);
%countEachLabel(imds_rand_Testsubset);

%% Flatten array for raw pixel classification
% CONFIRM THIS WORKS
X_train = reshape(cell2mat(imds_rand_Trainsubset.readall),[],3072);
X_test = reshape(cell2mat(imds_rand_Testsubset.readall),[],3072);

y_train = imds_rand_Trainsubset.Labels;
y_test = imds_rand_Testsubset.Labels;

X_train_val = X_train(1:size(X_train,1)*cv_train_split,:);
X_test_val = X_train(size(X_train,1)*cv_train_split+1:end,:);

y_train_val = y_train(1:size(y_train,1)*cv_train_split,:);
y_test_val = y_train(size(y_train,1)*cv_train_split+1:end,:);

%% CNN Feature Extractor

% Extract features from images using pretrained CNN
net = vgg16();

imageSize = net.Layers(1).InputSize;
augmentedTrain = augmentedImageDatastore(imageSize, imds_rand_Trainsubset);
augmentedTest = augmentedImageDatastore(imageSize, imds_rand_Testsubset);

% CHECK THAT fc6 IS CORRECT AND NOT POOL5 
X_train = activations(net,augmentedTrain,'fc6','OutputAs','rows');
X_test = activations(net,augmentedTest,'fc6','OutputAs','rows');

y_train = imds_rand_Trainsubset.Labels;
y_test = imds_rand_Testsubset.Labels;

X_train_val = X_train(1:size(X_train,1)*cv_train_split,:);
X_test_val = X_train(size(X_train,1)*cv_train_split+1:end,:);

y_train_val = y_train(1:size(y_train,1)*cv_train_split,:);
y_test_val = y_train(size(y_train,1)*cv_train_split+1:end,:);

%% Surf Feature Extractor
% Takes in training set and test set and returns feature vector for train
% validation and test set
% <CODE HERE>


% We then combine the three feature extraction aproaches into a function

%% SVM
classifier = fitcecoc(X_train,y_train);
YPred = classifier.predict(X_test);
avg_accuracy = mean(YPred == y_test);


%% Multilayer Perceptron - gridsearch for architecture

% CONFIRM THAT YOU NEED TO HAVE 10 AS THE OUTPUT LAYER
network_structure = {[4096,10];[100,100,100,100,100,100,100,100,10];
    [2048,1024,528,256,10]; [4096,2048,1024,10];
    [528,528,528,528,528,10]; [528,528,528,10];
    [256,528,1024,528,256,10]; [528,1024,528,10];
    [1024,528,256,528,1024,10]; [1024,528,1024,10];
    [100,100,100,10]; [32,10]; [32, 32,10]};

network_structure = {[528,528,10]; [100,100,100,10]; [32, 32, 10]}

MLP_architectures = zeros(numel(network_structure),4);

for i = 1:numel(network_structure)
    net = patternnet(network_structure{i}, 'traingdx');
    net.trainParam.epochs = 250;
    net.trainParam.time = 600;

    n_layers = size(net.layers,1)-1;
    net.layers{1:n_layers}.transferFcn = 'tansig';
    net.performParam.regularization = 0.1;
    net.performParam.normalization = 'none';

    net.input.processFcns = {'removeconstantrows','mapminmax'};
    net.divideFcn = 'dividerand';  % Divide data randomly
    net.divideMode = 'sample';  % Divide up every sample
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 30/100;
    net.performFcn = 'crossentropy';  % Cross-Entropy

    % Train the Network
    tic;
    [net,~] = train(net,double(X_train_val)',dummyvar(y_train_val)','useGPU','yes');

    % Training / Test Accuracy
    [~, preds] = max(net(double(X_train_val)'));
    train_accuracy = mean(y_train_val == categorical(categories(preds)'),'all');
    disp(['Training Accuracy: ', num2str(train_accuracy)])
    
    [~, preds] = max(net(double(X_test_val)'));
    test_accuracy = mean(y_test_val == categorical(categories(preds)'),'all');
    disp(['Test Accuracy: ', num2str(test_accuracy)])
    toc;

    elapsed_time = toc;
    
    MLP_architectures(i,:) = [i train_accuracy test_accuracy elapsed_time];
end

adjusted_perf = MLP_architectures(:,3) ./ MLP_architectures(:,4) *100;
MLP_architectures = [MLP_architectures, adjusted_perf];

disp('MLP Model Architecture Performance:')
disp(MLP_architectures)

%% MLP - gridsearch for model hyperparameters
% Select optimal architecture
optimal_architecture = network_structure{2}
regularization = [0.00;0.10;0.25;0.50];
learning_rate = [0.001;0.01;0.1];
count=0
%activation_function = {'tansig'; 'logsig'; 'poslin'};

MLP_hyperparameters = zeros(numel(regularization)*numel(learning_rate),6);

for i = 1:numel(regularization)
    net = patternnet(optimal_architecture, 'traingdx');
    net.trainParam.epochs = 250;
    net.trainParam.time = 600;

    n_layers = size(net.layers,1)-1;
    net.layers{1:n_layers}.transferFcn = 'tansig';
    net.performParam.regularization = regularization(i);
    net.performParam.normalization = 'none';
    
    for j = 1:numel(learning_rate)
        net.trainParam.lr = learning_rate(j);

        net.input.processFcns = {'removeconstantrows','mapminmax'};
        net.divideFcn = 'dividerand';  % Divide data randomly
        net.divideMode = 'sample';  % Divide up every sample
        net.divideParam.trainRatio = 70/100;
        net.divideParam.valRatio = 30/100;
        net.performFcn = 'crossentropy';  % Cross-Entropy

        % Train the Network
        tic;
        [net,~] = train(net,double(X_train_val)',dummyvar(y_train_val)','useGPU','yes');

       % Training / Test Accuracy
        [~, preds] = max(net(double(X_train_val)'));
        train_accuracy = mean(y_train_val == categorical(categories(preds)'),'all');
        disp(['Training Accuracy: ', num2str(train_accuracy)])

        [~, preds] = max(net(double(X_test_val)'));
        test_accuracy = mean(y_test_val == categorical(categories(preds)'),'all');
        disp(['Test Accuracy: ', num2str(test_accuracy)])
        toc;

        elapsed_time = toc;
        count = count + 1;
          
        MLP_hyperparameters(count,:) = [count regularization(i) learning_rate(j) ...
            train_accuracy test_accuracy elapsed_time];
    end
end

adjusted_perf = MLP_hyperparameters(:,5) ./ MLP_hyperparameters(:,6) *100;
MLP_hyperparameters = [MLP_hyperparameters, adjusted_perf];

disp('MLP Hyperparameter Performance:')
disp(MLP_hyperparameters)

%% MLP - Fit final model on whole train set and evaluate against test set 
% Uses optimal architecture and hyperparameters
% Update optimal parameter setting based on grid search
optimal_hyperparameters = 8;
optimal_regularization =  MLP_hyperparameters(optimal_hyperparameters,2);
optimal_learning_rate =  MLP_hyperparameters(optimal_hyperparameters,3);

net = patternnet(optimal_architecture, 'traingdx');
net.trainParam.epochs = 250;
net.trainParam.time = 600;

n_layers = size(net.layers,1)-1;
net.layers{1:n_layers}.transferFcn = 'tansig';
net.performParam.regularization = optimal_regularization;
net.performParam.normalization = 'none';
net.trainParam.lr = optimal_learning_rate;

net.input.processFcns = {'removeconstantrows','mapminmax'};
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 30/100;
net.performFcn = 'crossentropy';  % Cross-Entropy

% Train the Network
tic;
[net,tr] = train(net,double(X_train)',dummyvar(y_train)','useGPU','yes');

% Training / Test Accuracy
[~, preds] = max(net(double(X_train)'));
MLP_train_accuracy = mean(y_train == categorical(categories(preds)'),'all');
disp(['Training Accuracy: ', num2str(MLP_train_accuracy)])

[~, MLP_preds] = max(net(double(X_test)'));
MLP_test_accuracy = mean(y_test == categorical(categories(MLP_preds)'),'all');
disp(['Test Accuracy: ', num2str(MLP_test_accuracy)])
toc;

MLP_elapsed_time = toc;

disp('MLP Final Model Performance:')
disp(['Training Accuracy: ', num2str(MLP_train_accuracy)])
disp(['Test Accuracy: ', num2str(MLP_test_accuracy)])
disp(['Elapsed Time: ', num2str(MLP_elapsed_time)])
disp(['Adjusted Accuracy: ', num2str(MLP_test_accuracy / MLP_elapsed_time *100)])

%% Plot confusion matrices and evaluate performance

MLP_cm = confusionmat(y_test, categorical(categories(MLP_preds)'));

























