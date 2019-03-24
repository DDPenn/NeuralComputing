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

% Split into train/valid
[trainingSet, validationSet] = splitEachLabel(imds_rand_Trainsubset, 0.3, 'randomize');

% Extract BoF (SURF Feature extract) - then calc accuracy on validation
bag = bagOfFeatures(trainingSet);
trainFeatures = encode(bag, trainingSet);
SVM_SURF = fitcecoc(trainFeatures,trainingSet.Labels);
featureMatrix = encode(bag, validationSet);
[pred score cost] = predict(SVM_SURF, featureMatrix)
accuracy = sum(validationSet.Labels == pred)/size(validationSet.Labels,1);
accuracy; % 29%...

% Second approach, using imageTrainCatClassifier
bag = bagOfFeatures(trainingSet); % create bag of features from trainingSet (already run for the first approach..)
categoryClassifier = trainImageCategoryClassifier(trainingSet, bag);
confMatrix = evaluate(categoryClassifier, validationSet); % 27% avg accuracy


