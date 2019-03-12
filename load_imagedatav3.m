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
y_train = df(:,end);

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

