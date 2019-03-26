% %% Download the CIFAR-10 dataset
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

%%   

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

% Confirm partition worked and check for class imbalance
%countEachLabel(imds_rand_Trainsubset);
%countEachLabel(imds_rand_Testsubset);


%%

% Extract features from images using CNN
net = vgg16();

imageSize = net.Layers(1).InputSize;
augmentedTrain = augmentedImageDatastore(imageSize, imds_rand_Trainsubset);
augmentedTest = augmentedImageDatastore(imageSize, imds_rand_Testsubset);

% CHECK THAT fc6 IS CORRECT AND NOT POOL5 
X_train = activations(net,augmentedTrain,'fc6','OutputAs','rows');
X_test = activations(net,augmentedTest,'fc6','OutputAs','rows');

y_train = imds_rand_Trainsubset.Labels;
y_test = imds_rand_Testsubset.Labels;

% just save/export the working space so we do not need to repeat/upload
% anything everytime
% 
% save('

%% SVM
classifier = fitcecoc(X_train,y_train);
YPred = classifier.predict(X_test);
avg_accuracy = mean(YPred == y_test);


%% Multilayer Perceptron 
network_structure = {[4096,10];[100,100,100,100,100,100,100,100,10];
    [2048,1024,528,256,10]; [4096,2048,1024,10];
    [528,528,528,528,528,10]; [528,528,528,10];
    [256,528,1024,528,256,10]; [528,1024,528,10];
    [1024,528,256,528,1024,10]; [1024,528,1024,10];
    [100,100,100,10]}

network_structure = {[528,528,10]; [100,100,100,10]}

MLP_architectures = zeros(numel(network_structure),4)

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
    [net,tr] = train(net,double(X_train)',dummyvar(y_train)','useGPU','yes');

    % Training / Test Accuracy
    predict = round(net(double(X_train)'));
    train_accuracy = mean(dummyvar(y_train) == round(predict'),'all');
    disp(['Training Accuracy: ', num2str(train_accuracy)])

    predict = round(net(double(X_test)'));
    test_accuracy = mean(dummyvar(y_test) == round(predict'),'all');
    disp(['Test Accuracy: ', num2str(test_accuracy)])
    toc;

    elapsed_time = toc;
    
    MLP_architectures(i,:) = [i train_accuracy test_accuracy elapsed_time];
end

disp('MLP Model Architecture Performance:')
disp(MLP_architectures)

%%
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
        [net,tr] = train(net,double(X_train)',dummyvar(y_train)','useGPU','yes');

        % Training / Test Accuracy
        predict = round(net(double(X_train)'));
        train_accuracy = mean(dummyvar(y_train) == round(predict'),'all');
        disp(['Training Accuracy: ', num2str(train_accuracy)])

        predict = round(net(double(X_test)'));
        test_accuracy = mean(dummyvar(y_test) == round(predict'),'all');
        disp(['Test Accuracy: ', num2str(test_accuracy)])
        toc;

        elapsed_time = toc;
        count = count + 1;
          
        MLP_hyperparameters(count,:) = [count regularization(i) learning_rate(j) ...
            train_accuracy test_accuracy elapsed_time];
    end
end

disp('MLP Hyperparameter Performance:')
disp(MLP_hyperparameters)

%% Plot confusion matrices and evaluate performance

[~, max_idx] = max(predict);
MLP_cm = confusionmat(y_test, categorical(categories(max_idx)'));

























