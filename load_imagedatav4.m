%% Download the CIFAR-10 dataset
if ~exist('cifar-10-batches-mat','dir')
    cifar10Dataset = 'cifar-10-matlab';
    disp('Downloading 174MB CIFAR-10 dataset...');   
    websave([cifar10Dataset,'.tar.gz'],...
        ['https://www.cs.toronto.edu/~kriz/',cifar10Dataset,'.tar.gz']);
    gunzip([cifar10Dataset,'.tar.gz'])
    delete([cifar10Dataset,'.tar.gz'])
    untar([cifar10Dataset,'.tar'])
    delete([cifar10Dataset,'.tar'])
end

% Prepare the CIFAR-10 dataset
if ~exist('cifar10Train','dir')
    disp('Saving the Images in folders. This might take some time...');    
    saveCIFAR10AsFolderOfImages('cifar-10-batches-mat', pwd, true);
end

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
net = vgg16()

imageSize = net.Layers(1).InputSize;
augmentedTrain = augmentedImageDatastore(imageSize, imds_rand_Trainsubset)
augmentedTest = augmentedImageDatastore(imageSize, imds_rand_Testsubset)

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
%YPred = predict(classifier,X_test); %in case doesn t work-> YPred = classifier.predict(X_test);
YPred = classifier.predict(X_test);
avg_accuracy = mean(YPred == y_test); %78


%% Feed Forward Neural Network

net = patternnet([10,10], 'trainlm');
net.trainParam.epochs = 50;
n_layers = size(net.layers,1)-1;
net.layers{1:n_layers}.transferFcn = 'poslin';

net.input.processFcns = {'removeconstantrows','mapminmax'};

net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

net.performFcn = 'crossentropy';  % Cross-Entropy

net.performParam.regularization = 0.1;
net.performParam.normalization = 'none';

% Train the Network
[net,tr] = train(net,X_train',dummyvar(y_train)');

% Accuracy
predict = round(net(X_train'));
accuracy = sum(y_train == round(predict'))/length(y_train);

disp(['Accuracy: ', num2str(accuracy)])





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


%%

X_train_resized = [];
for i = 1:1000%size(X_train,4)
    resized = imresize(X_train(:,:,:,i),[224 224], 'nearest');
    X_train_resized = cat(4,X_train_resized,resized);
end

X_test_resized = [];
for i = 1:1000%size(X_train,4)
    resized = imresize(X_test(:,:,:,i),[224 224], 'nearest');
    X_test_resized = cat(4,X_test_resized,resized);
end






%% Neural Network https://uk.mathworks.com/help/deeplearning/examples/train-deep-learning-network-to-classify-new-images.html

net = vgg16()

imageSize = net.Layers(1).InputSize;
augmentedTest = augmentedImageDatastore(imageSize, X_train)

X_train_resized = [];
for i = 1:1000%size(X_train,4)
    resized = imresize(X_train(:,:,:,i),[224 224], 'nearest');
    X_train_resized = cat(4,X_train_resized,resized);
end

X_test_resized = [];
for i = 1:1000%size(X_train,4)
    resized = imresize(X_test(:,:,:,i),[224 224], 'nearest');
    X_test_resized = cat(4,X_test_resized,resized);
end

lgraph = layerGraph(net.Layers); 
findLayersToReplace(lgraph);

newLearnableLayer = fullyConnectedLayer(10, ...
        'Name','new_fc')

lgraph = replaceLayer(lgraph,'fc8',newLearnableLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'output',newClassLayer);

layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:32) = freezeWeights(layers(1:32));
lgraph = createLgraphUsingConnections(layers,connections);


% Figure out how to use these
%augimdsTrain = augmentedImageDatastore(inputSize(1:2),X_train);
%augimdsValidation = augmentedImageDatastore(inputSize(1:2),X_test);

net = trainNetwork(X_train_resized,y_train(1:1000),lgraph,options)
[ypred, probs] = classify(net,X_test_resized);
test_accuracy = mean(ypred == y_test(1:1000));

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');


trainNetwork(X_train_resized,y_train(1:1000),lgraph,options)


%feature extraction for specific layer which can be used in another model
featuresTrain = activations(net,X_train_resized,'fc6','OutputAs','rows');
featuresTest = activations(net,X_test_resized,'fc6','OutputAs','rows');

classifier = fitcecoc(featuresTrain,y_train(1:1000));
YPred = predict(classifier,featuresTest);
mean(YPred == y_test(1:1000));









layers = [
    imageInputLayer([32 32 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

%% 
df = readtable('pima-indians-diabetes.csv');
X_train = df(:,1:end-1);
X_train = table2array(X_train);

y_train = df(:,end);
y_train = table2array(y_train);

x = X_train';
t = y_train';

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainlm';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network
hiddenLayerSize = 69;
net = patternnet([500,50,20,10], trainFcn);


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

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotconfusion', 'plotroc'};

% Train the Network
[net,tr] = train(net,x,t);




%%

layers = [
    fullyConnectedLayer(10)
    fullyConnectedLayer(10)
    fullyConnectedLayer(10)
    
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(X_train, y_train, layers, options);

%% Flatten X_train, X_test dataset into obs/rows
X_train_flat = reshape(X_train,[],3072);
X_test_flat = reshape(X_test,[],3072);

% Standardize for SVM
X_train_std = double(X_train_flat)/255;
X_test_std = double(X_test_flat)/255;


%%
% Fit model and predict on train and test sets
svm_mdl = fitcecoc(X_train_std,y_train);

train_preds = svm_mdl.predict(X_train_std);
train_conf = confusionmat(y_train, train_preds); %it returns a 10x10 conf matrix
disp(['SVM Training Accuracy: ',num2str(sum(train_preds == y_train)/length(y_train))])

test_preds = svm_mdl.predict(X_test_std);
test_conf = confusionmat(y_test, test_preds);
disp(['SVM Test Accuracy: ',num2str(sum(test_preds == y_test)/length(y_test))])








































%%

load('imagedata.mat')

%%% Select a sample of the data to improve speed

X_train = trainingImages(:,:,:,1:5000);
y_train = trainingLabels(1:5000);
X_test = testImages(:,:,:,1:2500);
y_test = testLabels(1:5000);

% Examine the class distribution 
[class_count,label]=hist(y_train,unique(y_train));
disp(label)
disp(class_count/length(y_train))
clear('class_count', 'label')


% Convert y labels to one-hot encoding
%y_train = dummyvar(y_train);
%y_test = dummyvar(y_test);

% X_train_gray = [];
% for i = 1:size(X_train,4)
%     gray = rgb2gray(X_train(:,:,:,i));
%     X_train_gray = [X_train_gray;gray];
% end

