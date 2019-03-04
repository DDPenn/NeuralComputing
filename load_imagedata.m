load('imagedata.mat')

%%% it returns/create in the wordspace the train and test (img/labels)

X_train = trainingImages(:,:,:,1:10000);
y_train = trainingLabels(1:10000);
X_test = testImages(:,:,:,1:10000);
y_test = testLabels(1:10000);

%% reshaping X_train dataset into 10k obs/rows
X_t2 = reshape(X_train,[],3072);


%%reshape test also
X_test2 = reshape(X_test,[],3072);
 

%%% SVM %%%

% standardize X

X_t2_dub = double(X_t2);
X_std = X_t2_dub/255;

%std on test
X_test_dub = double(X_test2);
X_test_std = X_test_dub/255;


mdl_1 = fitcecoc(X_t2,y_train);

%%% just to have a little proof (we predicted on train)
pred = mdl_1.predict(X_std);
conf = confusionmat(y_train, pred); %it returns a 10x10 conf matrix
%%%% FCK OFF THIS ABOVE


%%%then predict on X_test
pred = mdl_1.predict(X_test_std);
conf = confusionmat(y_test, pred);

% 
% sum(pred == y_test)
% 
% ans =
% 
%    958 %---> so we have ~10% acc on test...

err = loss(mdl_1,X_test_std, y_test) %accuracy on tst
%%%% 