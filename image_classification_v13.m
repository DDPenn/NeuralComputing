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

rng(34); %set seed

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

% Set subset of total training and test data

train_samples = 2000;
test_samples = 500;

imds_rand_Trainsubset = splitEachLabel(imds_train,train_samples/10,'randomized');
imds_rand_Testsubset = splitEachLabel(imds_test,test_samples/10,'randomized');

imds_rand_Trainsubset = shuffle(imds_rand_Trainsubset);
imds_rand_Testsubset = shuffle(imds_rand_Testsubset);


%split imds_ramd_Train into train and valid (here our train = "train_val",
%while valid = "test_val")
[X_train_val, X_test_val] = splitEachLabel(imds_rand_Trainsubset, 0.7, 'randomize');

% Confirm partition worked and check for class imbalance
%countEachLabel(imds_rand_Trainsubset);
%countEachLabel(imds_rand_Testsubset);

% Manually set the feature extractor from Raw, CNN, or Surf
feature_extractor = 'Surf';

if isequal(feature_extractor, 'Raw')
    % Flatten array for raw pixel classification
    
    y_train_val = X_train_val.Labels;
    y_test_val = X_test_val.Labels;
    y_test = imds_rand_Testsubset.Labels;

    X_train_val = reshape(cell2mat(X_train_val.readall),[],3072);
    X_test_val = reshape(cell2mat(X_test_val.readall),[],3072);
    X_test = reshape(cell2mat(imds_rand_Testsubset.readall),[],3072);

    X_train_val = double(X_train_val);
    X_test_val = double(X_test_val);
    X_test = double(X_test);

elseif isequal(feature_extractor, 'CNN')
    % CNN Feature Extractor
    % Extract features from images using pretrained CNN
    net = vgg16();

    imageSize = net.Layers(1).InputSize;
    augmentedTrainVal = augmentedImageDatastore(imageSize, X_train_val);
    augmentedTestVal = augmentedImageDatastore(imageSize, X_test_val);
    augmentedTest = augmentedImageDatastore(imageSize, imds_rand_Testsubset);

    y_train_val = X_train_val.Labels;
    y_test_val = X_test_val.Labels;
    y_test = imds_rand_Testsubset.Labels;

    % CHECK THAT fc6 IS CORRECT AND NOT POOL5 
    X_train_val = activations(net,augmentedTrainVal,'fc6','OutputAs','rows');
    X_test_val = activations(net,augmentedTestVal,'fc6','OutputAs','rows');
    X_test = activations(net,augmentedTest,'fc6','OutputAs','rows');

    X_train_val = double(X_train_val);
    X_test_val = double(X_test_val);
    X_test = double(X_test);

elseif isequal(feature_extractor, 'Surf')
    % Surf Feature Extractor
    bag = bagOfFeatures(X_train_val); %BoF on train

    y_train_val = X_train_val.Labels;
    y_test_val = X_test_val.Labels;
    y_test = imds_rand_Testsubset.Labels;

    X_train_val = encode(bag, X_train_val);
    X_test_val = encode(bag, X_test_val);
    X_test = encode(bag, imds_rand_Testsubset);

    X_train_val = double(X_train_val);
    X_test_val = double(X_test_val);
    X_test = double(X_test);

else
    disp("Please select a valid feature extractor: Raw, CNN, or Surf")
end

% Standardize the feature vectors
if isequal(feature_extractor, 'Raw')
    X_train_val = X_train_val / 255;
    X_test_val = X_test_val / 255;
    X_test = X_test / 255;
end

% Normalize
if isequal(feature_extractor, 'Surf') || isequal(feature_extractor, 'CNN')
    X_train_val = normalize(X_train_val);
    X_test_val = normalize(X_test_val);
    X_test = normalize(X_test);   
end

% Merge transformed validation sets into full training set 
X_train = [X_train_val;X_test_val];
y_train = [y_train_val;y_test_val];

%% Multilayer Perceptron - gridsearch for architecture

% CONFIRM THAT YOU NEED TO HAVE 10 AS THE OUTPUT LAYER
network_structure = {[4096,10];[100,100,100,100,100,100,100,100,10];
    [2048,1024,528,256,10]; [4096,2048,1024,10];
    [528,528,528,528,528,10]; [528,528,528,10];
    [256,528,1024,528,256,10]; [528,1024,528,10];
    [1024,528,256,528,1024,10]; [1024,528,1024,10];
    [100,100,100,10]; [32,10]; [32, 32,10]};

network_structure = {[528,528,10]; [100,100,100,10]; [32, 32, 10]};

MLP_architectures = zeros(numel(network_structure),4);

for i = 1:numel(network_structure)
    net = patternnet(network_structure{i}, 'traingdx');
    net.trainParam.epochs = 250;
    net.trainParam.time = 600;
    net.trainParam.max_fail = 10;

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
    [net,~] = train(net,X_train_val',dummyvar(y_train_val)','useGPU','yes');

    % Training / Test Accuracy
    [~, preds] = max(net(X_train_val'));
    train_accuracy = mean(y_train_val == categorical(categories(preds)'),'all');
    disp(['Training Accuracy: ', num2str(train_accuracy)])
    
    [~, preds] = max(net(X_test_val'));
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
optimal_architecture = network_structure{2};
regularization = [0.00;0.10;0.25;0.50];
learning_rate = [0.001;0.01;0.1];
count=0;
%activation_function = {'tansig'; 'logsig'; 'poslin'};

MLP_hyperparameters = zeros(numel(regularization)*numel(learning_rate),6);

for i = 1:numel(regularization)
    net = patternnet(optimal_architecture, 'traingdx');
    net.trainParam.epochs = 250;
    net.trainParam.time = 600;
    net.trainParam.max_fail = 10;

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
        [net,~] = train(net,X_train_val',dummyvar(y_train_val)','useGPU','yes');

       % Training / Test Accuracy
        [~, preds] = max(net(X_train_val'));
        train_accuracy = mean(y_train_val == categorical(categories(preds)'),'all');
        disp(['Training Accuracy: ', num2str(train_accuracy)])

        [~, preds] = max(net(X_test_val'));
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
optimal_hyperparameters = 5;
optimal_regularization =  MLP_hyperparameters(optimal_hyperparameters,2);
optimal_learning_rate =  MLP_hyperparameters(optimal_hyperparameters,3);

net = patternnet(optimal_architecture, 'traingdx');
net.trainParam.epochs = 250;
net.trainParam.time = 600;
net.trainParam.max_fail = 10;

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
[net,tr] = train(net,X_train',dummyvar(y_train)','useGPU','yes');

% Training / Test Accuracy
[~, preds] = max(net(X_train'));
MLP_train_accuracy = mean(y_train == categorical(categories(preds)'),'all');
disp(['Training Accuracy: ', num2str(MLP_train_accuracy)])

[~, MLP_preds] = max(net(X_test'));
MLP_test_accuracy = mean(y_test == categorical(categories(MLP_preds)'),'all');
disp(['Test Accuracy: ', num2str(MLP_test_accuracy)])
toc;

MLP_elapsed_time = toc;

disp('MLP Final Model Performance:')
disp(['Training Accuracy: ', num2str(MLP_train_accuracy)])
disp(['Test Accuracy: ', num2str(MLP_test_accuracy)])
disp(['Elapsed Time: ', num2str(MLP_elapsed_time)])
disp(['Adjusted Accuracy: ', num2str(MLP_test_accuracy / MLP_elapsed_time *100)])
[MLP_train_accuracy MLP_test_accuracy MLP_elapsed_time MLP_test_accuracy / MLP_elapsed_time *100];

%% Plot confusion matrices and evaluate performance

MLP_cm = confusionmat(y_test, categorical(categories(MLP_preds)'));

plotconfusion(categorical(categories(MLP_preds)'), y_test,'MLP Confusion Matrix');


%% Fit SVM Model

kernels = {'linear','gaussian','polynomial'}; %'gaussian', 'polynomial'

SVM_hyperparameters = zeros(numel(kernels),4);
count = 0;

for i = 1:numel(kernels)
    
    tic;
    t = templateSVM('KernelFunction', kernels{i}); % fitcecoc requires an SVM template
    svm = fitcecoc(X_train_val, y_train_val, 'learners', t);
    preds = svm.predict(X_train_val);
    train_accuracy = mean(preds == y_train_val);
    
    preds = svm.predict(X_test_val);
    test_accuracy = mean(preds == y_test_val);
    
    toc;
    
    elapsed_time = toc;
    count = count + 1;
    
    SVM_hyperparameters(count,:) = [i ...
            train_accuracy test_accuracy elapsed_time];
end

adjusted_perf = SVM_hyperparameters(:,3) ./ SVM_hyperparameters(:,4) *100;
SVM_hyperparameters = [SVM_hyperparameters, adjusted_perf];

disp('SVM Hyperparameter Performance:')
disp(SVM_hyperparameters)

%% SVM - gridsearch for model hyperparameters
% Select optimal SVM model

optimal_kernel = kernels{1};% ,'linear'{1}; %'gaussian'{2}, 'polynomial'{3}
boxconst = [0.01;0.10;0.3;0.5;1];
coding =strcat({'onevsone','onevsall'});

%solution suggested
SVM_hyperparameters_table = cell2table(cell(0,6)); %initialize an empy tab
% SVM_hyperparameters = zeros(numel(boxconst)*numel(coding),6);
count = 0;

for i = 1:numel(boxconst)
    for j = 1:numel(coding)
        
        tic;
        t = templateSVM('KernelFunction', optimal_kernel, 'BoxConstraint', boxconst(i));
        svm = fitcecoc(X_train_val, y_train_val, 'learners', t, 'Coding', coding{j});
        preds = svm.predict(X_train_val);
        train_accuracy = mean(preds == y_train_val);
%         disp(i) %can comment if it all works
%         disp(j) %can comment if it all works
    
        preds = svm.predict(X_test_val);
        test_accuracy = mean(preds == y_test_val);

        toc;
    
        elapsed_time = toc;
        count = count + 1;
        a = horzcat(count,boxconst(i),train_accuracy ,test_accuracy ,elapsed_time,cellstr(coding(j)));
        SVM_hyperparameters_table=[SVM_hyperparameters_table;a];
%         SVM_hyperparameters(count,:) = [count boxconst(i) cellstr(coding(j)) ...
%             train_accuracy test_accuracy elapsed_time];
    end
end

adjusted_perf_SVM = SVM_hyperparameters_table{:,4}./SVM_hyperparameters_table{:,5}*100; %IT WORKS! CHECK IT OUT, ANDREW
SVM_hyperparameters_table = [SVM_hyperparameters_table, table(adjusted_perf_SVM)];

disp('SVM Hyperparameter Performance - 2nd Level:')
disp(SVM_hyperparameters_table)

%% SVM - Fit final model on whole train set and evaluate against test set 
% Uses optimal kernel and hyperparameters
% Update optimal parameter setting based on grid search
optimal_kernel=kernels{1};
optimal_boxconst = 1; %for raw data is 0.01, else is 1
optimal_coding = 'onevsone'; %for raw data is onevsall, whereas is onevsone

% Train the classifier
tic;
t = templateSVM('KernelFunction', optimal_kernel, 'BoxConstraint', optimal_boxconst);
svm = fitcecoc(X_train, y_train, 'learners', t, 'Coding', optimal_coding);
preds = svm.predict(X_train);
SVM_train_accuracy = mean(preds == y_train);
    
%Testing on Final Test Set
SVM_preds = svm.predict(X_test);
SVM_test_accuracy = mean(SVM_preds == y_test);
    
toc;
   
SVM_elapsed_time = toc;

% displaying results
disp('SVM Final Model Performance:')
disp(['Training Accuracy: ', num2str(SVM_train_accuracy)])
disp(['Test Accuracy: ', num2str(SVM_test_accuracy)])
disp(['Elapsed Time: ', num2str(SVM_elapsed_time)])
disp(['Adjusted Accuracy: ', num2str(SVM_test_accuracy / SVM_elapsed_time *100)])

[SVM_train_accuracy SVM_test_accuracy SVM_elapsed_time SVM_test_accuracy / SVM_elapsed_time *100] %final answer/array with SVM results

%% Plot confusion matrices and evaluate performance

SVM_cm = confusionmat(y_test, SVM_preds);
plotconfusion(SVM_preds, y_test, 'SVM Confusion Matrix');