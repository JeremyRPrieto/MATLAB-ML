close all;
clearvars;
clc;

addpath('C:\Users\jeremyprieto31\OneDrive\Documents\Graduate\MATLAB Toolboxes')

%% Load in Lead Datasets

% Adds directories to path if they exist
if exist('Leads_Hyp','dir'); addpath Leads_Hyp; end
if exist('Leads_NonHyp','dir'); addpath Leads_NonHyp; end
if exist('Leads','dir'); addpath Leads; end

folderInfo = dir('Leads');

for files = 3:length(folderInfo)
    filename = folderInfo(files).name;
    load(filename)
end

load responseVars.mat
ogTestData = cell2mat([responseVar_NH(2:end,:);responseVar_H(2:end,:)]);

%% Model Data Creation

seed1 = 13; % Seed for picking random features/lead combos
featuresTot = 29; % Total features

rng(seed1);
numRandPerms = 16;

for class = 1:5
    for lead = 0:15
        
        eval(['tempLead = Lead' num2str(lead+1) ';']);
        responseVars = [responseVar_NH;responseVar_H(2:end,:)];
        responseClass = responseVars;
        
        classVars = cell2mat(responseVars(2:end,1)) == class;
        
        for vars = 1:size(responseVars,1)-1
            if ~classVars(vars)
                responseClass{vars+1,1} = 0;
            else
                responseClass{vars+1,1} = 1;
            end
        end
        
        for rnd = 1:numRandPerms
            randFeats = randperm(featuresTot,15);
            
            tempData = tempLead(2:end,sort(randFeats));
            tempFeats = tempLead(1,sort(randFeats));
            
            modelData{rnd+(lead*numRandPerms),1} = [tempData responseClass(2:end,:) responseVars(2:end,1)];
            featNames{rnd+(lead*numRandPerms)}(1,:) = tempFeats;
            featNames{rnd+(lead*numRandPerms)}(2,:) = num2cell(sort(randFeats));
            
        end
    end
    modelDataOVA{class} = modelData;
    featDataOVA{class} = featNames';
end

rng(0);

close all;
clearvars -except modelDataOVA featDataOVA ogTestData;
clc;

%% Model Creation for Datasets
seed2 = 31;
rng(seed2);
testSubNum = randperm(46,2);
testSub1 = testSubNum(1); testSub2 = testSubNum(2);
% testSubNum = randperm(46,1);
ogYtest = ogTestData(ogTestData(:,2) == testSub1 | ogTestData(:,2) == testSub2,1);
% ogYtest = ogTestData(ogTestData(:,2) == testSubNum,1);

for class = 1:5
    disp([newline 'Running Models for Class: ' num2str(class) newline])
    
    tClass = tic;
    
    mdlClassData = modelDataOVA{class};
    
    for data = 1:length(mdlClassData)
        
        tic
        
        tempData = cell2mat(mdlClassData{data});
        
%         XtestData{data} = tempData(tempData(:,17) == testSub1 | tempData(:,17) == testSub2,1:15);
%         YtestData{data} = categorical(tempData(tempData(:,17) == testSub1 | tempData(:,17) == testSub2,16));
%         
%         tempData = tempData(tempData(:,17) ~= testSub1 & tempData(:,17) ~= testSub2,:);
%         subNums = unique(tempData(:,17));
%         
%         parfor kFold = 1:length(subNums)
%             
%             X_val = tempData(tempData(:,17) == subNums(kFold),1:15); Y_val = categorical(tempData(tempData(:,17) == subNums(kFold),16));
%             
%             totData = tempData(tempData(:,17) ~= subNums(kFold),1:15);
%             respVars = tempData(tempData(:,17) ~= subNums(kFold),16:end);
%             
%             [X,synthClass] = smote(totData,[],'Class',respVars(:,1));
%             
%             synthData = [X synthClass];
%             
%             synthData = synthData(randperm(size(synthData,1)),:);
%             
%             X_train = synthData(:,1:end-1); Y_train = categorical(synthData(:,end));
%             
%             Mdls = TreeBagger(150,X_train,Y_train,'OOBPredictorImportance','On',...
%                 'PredictorSelection','curvature','Method','classification','MinLeafSize',2);
%             
%             [labels{kFold},scores{kFold}] = predict(Mdls,X_val);
%             
%             C{kFold} = confusionmat(categorical(labels{kFold}),Y_val);
%             
%             [results{kFold},~] = getValues(C{kFold});
%             
%         end
        
        cvPart = cvpartition(size(tempData,1),'KFold',10);
        
        numFolds = cvPart.NumTestSets;
        
        for kFold = 1:numFolds
        
            X_test = tempData(cvPart.test(kFold),1:15); Y_test = categorical(tempData(cvPart.test(kFold),16));
            
            totData = tempData(cvPart.training(kFold),1:15);
            respVars = tempData(cvPart.training(kFold),16:end);
            
            [X,synthClass] = smote(totData,[],'Class',respVars(:,1));
            
            synthData = [X synthClass];
            
            synthData = synthData(randperm(size(synthData,1)),:);
            
            X_train = synthData(:,1:end-1); Y_train = categorical(synthData(:,end));
            
%             Mdls = TreeBagger(150,X_train,Y_train,'OOBPredictorImportance','On',...
%                 'PredictorSelection','curvature','Method','classification','MinLeafSize',2);
            
            [labels{kFold},scores{kFold}] = predict(Mdls,X_test);
            
            C{kFold} = confusionmat(categorical(labels{kFold}),Y_test,'Order',{'1','0'});
            
            [results{kFold},~] = getValues(C{kFold});
            
        end
        
        toc
        
        labelsData{data} = labels;
        scoresData{data} = scores;
        CData{data} = C;
        resultsData{data} = results;
        
        clear labels scores C results
        
%         %% Test Data for 5 Class %%%
%         X_test{data} = tempData(tempData(:,17) == testSub1 | tempData(:,17) == testSub2,1:15);
% %         X_test{data} = tempData(tempData(:,17) == testSubNum,1:15); 
%         Y_test{data} = tempData(tempData(:,17) == testSub1 | tempData(:,17) == testSub2,16);
% %         Y_test{data} = tempData(tempData(:,17) == testSubNum,16);
%         Y_test{data} = categorical(Y_test{data});
%         
%         %%% Generate Training Data Sets
% %         tempData = tempData(randperm(size(tempData, 1)), :);
%         totData = tempData(tempData(:,17) ~= testSub1 & tempData(:,17) ~= testSub2,1:15);
% %         totData = tempData(tempData(:,17) ~= testSubNum,1:15);
%         respVars = tempData(tempData(:,17) ~= testSub1 & tempData(:,17) ~= testSub2,16:end);
% %         respVars = tempData(tempData(:,17) ~= testSubNum,16:end);
%         
%         [trainData,valData,~] = dividerand(size(totData,1),1,0,0);
%         
%         [X,synthClass] = smote(totData(trainData,:),[],'Class',respVars(trainData,1));
%         
%         synthData = [X synthClass];
%         
%         synthData = synthData(randperm(size(synthData, 1)), :);
%         
%         X_train{data} = synthData(:,1:end-1); Y_train{data} = categorical(synthData(:,end));
% %         X_val{data} = totData(valData,:); Y_val{data} = categorical(respVars(valData,1));
%         
%         %%% Train data for 5 CLASS %%%
%         % X_train{data} = totData; Y_train{data} = respVars(:,1); Y_train{data} = categorical(Y_train{data});
%         
%         Mdls{data} = TreeBagger(150,X_train{data},Y_train{data},'OOBPredictorImportance','On',...
%             'PredictorSelection','curvature','Method','classification','MinLeafSize',2);
        
%         toc
        
    end
    
    tEnd = toc(tClass);
    
    disp([newline 'Program Time for Class: ' num2str(tEnd) ' seconds' newline])
    
    labelsOVA{class} = labelsData;
    scoresOVA{class} = scoresData;
    confOVA{class} = CData;
    resultsOVA{class} = resultsData;
    
    clear labelsData scoresData CData resultsData
    
%     Mdls_OVA{class,1} = Mdls;
%     X_testOVA{class,1} = X_test;
%     Y_testOVA{class,1} = Y_test;
%     X_valOVA{class,1} = X_val;
%     Y_valOVA{class,1} = Y_val;
%     X_trainOVA{class,1} = X_train;
%     Y_trainOVA{class,1} = Y_train;
    
end

%% KFold Adaption for per Subject

numSubs = 46;

for kfold = 1:numSubs
    
    disp([newline 'Running Models for K-Fold: ' num2str(kfold) newline])
    
    tClass = tic;
    
    for class = 1:5
       
        mdlClassData = modelDataOVA{class};
        
        tic
        
        parfor model = 1:length(mdlClassData)
            
            modelData = cell2mat(mdlClassData{model});

            X_test = modelData(modelData(:,17) == kfold,1:15);
            Y_test = categorical(modelData(modelData(:,17) == kfold,16));
            trueLabelTest = modelData(modelData(:,17) == kfold,end);
            
            tempData = modelData(modelData(:,17) ~= kfold,1:16);
            
            [X,synthClass] = smote(tempData(:,1:15),[],'Class',tempData(:,16));
            
            synthData = [X synthClass];
            
            synthData = synthData(randperm(size(synthData,1)),:);
            
            X_train = synthData(:,1:end-1); Y_train = categorical(synthData(:,end));
            
            Mdls = TreeBagger(150,X_train,Y_train,'PredictorSelection','curvature',... 
                'Method','classification','MinLeafSize',2);
            
            [labels{model},scores{model}] = predict(Mdls,X_test);
            
            C{model} = confusionmat(categorical(labels{model}),Y_test,'Order',{'1','0'});
            
            [results{model},~] = getValues(C{model});
            
            testData{model} = X_test;
            testLabels{model} = trueLabelTest;
            
        end
        
        toc
        
        labelsClass{class} = labels;
        scoresClass{class} = scores;
        CClass{class} = C;
        resultsClass{class} = results;
        testDataClass{class} = testData;
        testLabelsClass{class} = testLabels;
        
        clear labels scores C results testData testLabels
        
    end
    
    tEnd = toc(tClass);
    
    disp([newline 'Program Time for K-Fold: ' num2str(tEnd) ' seconds' newline])
    
    labelsKFold{class} = labelsClass;
    scoresKFold{class} = scoresClass;
    CKFold{class} = CClass;
    resultsKFold{class} = resultsClass;
    testDataKFold{class} = testDataClass;
    testLabelsKFold{class} = testLabelsClass;
    
    clear labelsClass scoresClass CClass resultsClass testDataClass testLabelsClass
    
end

%% Running through the models made per class

% for class = 1:size(Mdls_OVA,1)
%     
%     disp([newline 'Running Models for Class: ' num2str(class) newline])
%     
%     classMdls = Mdls_OVA{class};
%     classXtest = X_testOVA{class};
%     classYtest = Y_testOVA{class};
%     classXtrain = X_trainOVA{class};
%     classYtrain = Y_trainOVA{class};
% %     classXval = X_valOVA{class};
% %     classYval = Y_valOVA{class};
%     
%     parfor model = 1:size(classMdls,2)
%         
%         tic
%         
%         tempMdl = classMdls{model};
%         tempXtest = classXtest{model};
%         tempYtest = classYtest{model};
%         
%         tempXtrain = classXtrain{model};
%         tempYtrain = classYtrain{model};
%         
% %         tempXval = classXval{model};
% %         tempYval = classYval{model};
%         
%         [labels{model},score{model}] = predict(tempMdl,tempXtest);
%         
%         C{model} = confusionmat(categorical(labels{model}),tempYtest);
%         
%         [results{model},~] = getValues(C{model});
%             
%         toc
%         
%     end
%     
%     labelClass{class} = labels; scoreClass{class} = score;
%     resultsClass{class} = results; confClass{class} = C;
%     
%     clear labels score C results
%     
% end

disp([newline 'All models have been ran' newline])

%% Put together all results for clear graphic interpretation

for class = 1:size(resultsOVA,2)
    for model = 1:size(resultsOVA{class},2)
        statsAll((class-1)*size(resultsOVA{class},2)+model,1) = resultsOVA{class}{model}.Accuracy;
        statsAll((class-1)*size(resultsOVA{class},2)+model,2) = resultsOVA{class}{model}.F1_score;
        statsAll((class-1)*size(resultsOVA{class},2)+model,3) = resultsOVA{class}{model}.MatthewsCorrelationCoefficient;
        
        stats{class}(model,1) = resultsOVA{class}{model}.Accuracy;
        stats{class}(model,2) = resultsOVA{class}{model}.F1_score;
        stats{class}(model,3) = resultsOVA{class}{model}.MatthewsCorrelationCoefficient;
    end
end

%% Grab best model and predict on X test to see which class should be chosen

% for class = 1:size(resultsOVA,2)
%     [tempRank,indRank] = sort(stats{class}(:,2));
%     bestMdl{class,1} = Mdls_OVA{class}(indRank(end-14:end));
%     bestMdl{class,2} = indRank(end-14:end);
% end

for class = 1:size(bestMdl,1)
    disp([newline 'Running Scores for Class: ' num2str(class) newline])
    for model = 1:size(bestMdl{class},2)
        tic
        
        [labelsOVA{class}(:,model),scoresOVA{class,model}] = predict(bestMdl{class,1}{model},X_testOVA{class}{bestMdl{class,2}(model)});
        Cmats{class,model} = confusionmat(categorical(labelsOVA{class}(:,model)),Y_testOVA{class}{bestMdl{class,2}(model)});
        [resultsOVA{class,model},~] = getValues(Cmats{class,model});
        
        toc
    end
end

for class = 1:size(scoresOVA,1)
    for model = 1:size(scoresOVA,2)
        if model == 1
            combScoresOVA{class} = scoresOVA{class,model}(:,2);
        else
            combScoresOVA{class} = [combScoresOVA{class} scoresOVA{class,model}(:,2)];
        end
%         
%         if class == 1
%             combScoresOVA = scoresOVA{class}(:,2);
%         else
%             combScoresOVA = [combScoresOVA scoresOVA{class}(:,2)];
%         end
    end
end

for class = 1:length(combScoresOVA)
    scoresTot(:,class) = CentroidMethodGen(combScoresOVA{class});
    scoresMean(:,class) = mean(combScoresOVA{class},2);
end
    
[~,finalPredicts] = max(scoresTot,[],2);

C_test = confusionmat(finalPredicts,ogYtest)
resFinal = getValues(C_test)

for i = 1:length(combScoresOVA)
    bestAUCs(i) = multiClassAUC(combScoresOVA{i},ogYtest);
end

AUC = multiClassAUC(scoresMean,ogYtest)

save('FinalOutputsOVA.mat','finalPredicts','C_test','resFinal','AUC','bestAUCs','bestMdl','stats','statsAll',...
    'ogYtest','combScoresOVA','labelsOVA','scoresOVA','Cmats','resultsOVA','-v7.3')

%%

for i = 1:length(finalPredicts)
   if finalPredicts(i) == 1
       finalPredictsStr{i} = 'TP';
   elseif finalPredicts(i) == 2
       finalPredictsStr{i} = 'TA';
   elseif finalPredicts(i) == 3
       finalPredictsStr{i} = 'TE';
   elseif finalPredicts(i) == 4
       finalPredictsStr{i} = 'TR';
   elseif finalPredicts(i) == 5
       finalPredictsStr{i} = 'OTH';
   end
end

for i = 1:length(ogYtest)
   if ogYtest(i) == 1
       ogYtestStr{i} = 'TP';
   elseif ogYtest(i) == 2
       ogYtestStr{i} = 'TA';
   elseif ogYtest(i) == 3
       ogYtestStr{i} = 'TE';
   elseif ogYtest(i) == 4
       ogYtestStr{i} = 'TR';
   elseif ogYtest(i) == 5
       ogYtestStr{i} = 'OTH';
   end
end

%%
for class = 1:size(confOVA,2)
    for model = 1:size(confOVA{class},2)
        for kFold = 1:size(confOVA{class}{model},2)
            
            if kFold == 1
                sumConfOVA{class}{model} = confOVA{class}{model}{kFold};
            else
                sumConfOVA{class}{model} = sumConfOVA{class}{model} + confOVA{class}{model}{kFold};
            end
            
        end
    end
end

for class = 1:size(sumConfOVA,2)
    for model = 1:size(sumConfOVA{class},2)
        
        confMat = sumConfOVA{class}{model};
        
        precision = confMat(2,2) / (confMat(2,2) + confMat(1,2));
        recall = confMat(2,2) / (confMat(2,2) + confMat(2,1));
        specificity = confMat(1,1) / (confMat(1,1) + confMat(1,2));
        
        accuracy = (confMat(1,1) + confMat(2,2)) / (confMat(1,1) + confMat(2,2) + confMat(1,2) + confMat(2,1));
        f1_score = 2 * ((precision * recall) / (precision + recall));
        auc = trapz([0 specificity 1],[0 recall 1]);
        
        stats{class}{model,1} = accuracy;
        stats{class}{model,2} = f1_score;
        stats{class}{model,3} = auc;
        
    end
end

total = 0;

for i = 1:46
    x = sum(tempData(:,17) == i);
    total = total + x;
    disp(['Sub Num: ' num2str(i) ', Sum: ' num2str(x)])

end

disp(['Total: ' num2str(total)])