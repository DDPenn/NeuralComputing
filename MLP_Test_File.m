load('MLP_Model_env')

categories = {'airplane', 'automobile', 'bird', 'cat',...
                'deer','dog','frog','horse','ship','truck'};

tic;
[~, MLP_preds] = max(net(X_test'));
MLP_test_accuracy = mean(y_test == categorical(categories(MLP_preds)'),'all');
toc;

MLP_elapsed_time = toc;

disp('MLP Final Model Performance:')
disp(['Test Accuracy: ', num2str(MLP_test_accuracy)])
disp(['Elapsed Time: ', num2str(MLP_elapsed_time)])
disp(['Adjusted Accuracy: ', num2str(MLP_test_accuracy / MLP_elapsed_time *100)])

MLP_cm = confusionmat(y_test, categorical(categories(MLP_preds)'));
plotconfusion(categorical(categories(MLP_preds)'), y_test,'MLP Confusion Matrix');

