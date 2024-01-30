close all;
clearvars -except Mdls_OVA;
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

leads = 16; K = 2;

P = nchoosek(1:leads,K);
% P=reshape(P(:,perms(1:K)),[],K);

seed1 = 13; % Seed for picking random features/lead combos
featuresTot = 29; % Total features
numRandPerms = 3;

rng(seed1);

randPermList = randperm(size(P,1));

P = P(randPermList,:);

for perm = 1:size(P,1)
    
    lead1 = P(perm,1); lead2 = P(perm,2);
    eval(['tempLead1 = Lead' num2str(lead1) ';']);
    eval(['tempLead2 = Lead' num2str(lead2) ';']);
    
    responseVars = [responseVar_NH;responseVar_H(2:end,:)];
    
    for rnd = 1:numRandPerms
        randFeats = randperm(featuresTot,15);
    
        modelDataComOVA{rnd+((perm-1)*numRandPerms),1} = [tempLead1(:,sort(randFeats)) tempLead2(:,sort(randFeats)) responseVars];
        featNamesComOVA{rnd+((perm-1)*numRandPerms)}(1,:) = modelDataComOVA{rnd+((perm-1)*numRandPerms),1}(1,1:15*2);
        featNamesComOVA{rnd+((perm-1)*numRandPerms)}(2,:) = num2cell([sort(randFeats) sort(randFeats)]);
    end
    
end

for class = 1:5
    for perm = 1:size(modelDataComOVA,1)
        
        tic
    
        tempData = modelDataComOVA{perm}(2:end,:);
        respVars = tempData(:,31);
        
        classVars = cell2mat(tempData(:,31)) == class;
        
        for vars = 1:size(classVars,1)
            if ~classVars(vars)
                respVars{vars,1} = 0;
            else
                respVars{vars,1} = 1;
            end
        end
        
        tempData(:,31) = respVars;
        
        modelData{perm,class} = tempData;
        
        toc
    end
end

rng(0);


%% Model Creation

close all;
clearvars -except modelData ogTestData Mdls_OVA;
clc;

seed2 = 31;
rng(seed2);
testSubNum = randperm(46,2);
testSub1 = testSubNum(1); testSub2 = testSubNum(2);

ogYtest = ogTestData(ogTestData(:,2) == testSub1 | ogTestData(:,2) == testSub2,1);

for class = 1:5
    disp([newline 'Running Models for Class: ' num2str(class) newline])
    
    tClass = tic;
    
    for data = 1:size(modelData,1)
        
        tic
        
        tempData = cell2mat(modelData{data,class});
        cvPart = cvpartition(size(tempData,1),'KFold',10);
        
        numFolds = cvPart.NumTestSets;
        
        parfor kFold = 1:numFolds
            
            X_test = tempData(cvPart.test(kFold),1:30); Y_test = categorical(tempData(cvPart.test(kFold),31));
            
            totData = tempData(cvPart.training(kFold),1:30);
            respVars = tempData(cvPart.training(kFold),31:end);
            
            [X,synthClass] = smote(totData,[],'Class',respVars(:,1));
            
            synthData = [X synthClass];
            
            synthData = synthData(randperm(size(synthData,1)),:);
            
            X_train = synthData(:,1:end-1); Y_train = categorical(synthData(:,end));
            
            Mdls = TreeBagger(150,X_train,Y_train,'OOBPredictorImportance','On',...
                'PredictorSelection','curvature','Method','classification','MinLeafSize',2);
            
            [labels{kFold},scores{kFold}] = predict(Mdls,X_test);
            
            C{kFold} = confusionmat(categorical(labels{kFold}),Y_test);
            
            [results{kFold},~] = getValues(C{kFold});
            
        end
        
        toc
        
        labelsData{data} = labels;
        scoresData{data} = scores;
        CData{data} = C;
        resultsData{data} = results;
        
        clear labels scores C results
        
%         %%% Test Data for 5 Class %%%
%         X_test{data} = tempData(tempData(:,32) == testSub1 | tempData(:,32) == testSub2,1:30);
% %         X_test{data} = tempData(tempData(:,17) == testSubNum,1:15); 
%         Y_test{data} = tempData(tempData(:,32) == testSub1 | tempData(:,32) == testSub2,31);
% %         Y_test{data} = tempData(tempData(:,17) == testSubNum,16);
%         Y_test{data} = categorical(Y_test{data});
%         
%         %%% Generate Training Data Sets
% %         tempData = tempData(randperm(size(tempData, 1)), :);
%         totData = tempData(tempData(:,32) ~= testSub1 & tempData(:,32) ~= testSub2,1:30);
% %         totData = tempData(tempData(:,17) ~= testSubNum,1:15);
%         respVars = tempData(tempData(:,32) ~= testSub1 & tempData(:,32) ~= testSub2,31:end);
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
%         X_train = synthData(:,1:end-1); Y_train = categorical(synthData(:,end));
% %         X_val{data} = totData(valData,:); Y_val{data} = categorical(respVars(valData,1));
%         
%         %%% Train data for 5 CLASS %%%
%         % X_train{data} = totData; Y_train{data} = respVars(:,1); Y_train{data} = categorical(Y_train{data});
%         
%         Mdls{data} = TreeBagger(150,X_train,Y_train,'OOBPredictorImportance','On',...
%             'PredictorSelection','curvature','Method','classification','MinLeafSize',2);
%         
%         [labels{data},score{data}] = predict(Mdls{data},X_test{data});
%         
%         C{data} = confusionmat(categorical(labels{data}),Y_test{data});
%         
%         [results{data},~] = getValues(C{data});
%         
%         toc
        
    end
    
    tEnd = toc(tClass);
    
    disp([newline 'Program Time for Class: ' num2str(tEnd) ' seconds' newline])
    
    labelClass{class} = labels; scoreClass{class} = score;
    resultsClass{class} = results; confClass{class} = C;
    
%     Mdls_OVA{class,1} = Mdls;
%     X_testOVA{class,1} = X_test;
%     Y_testOVA{class,1} = Y_test;
    
    clear labels score C results Mdls
    
    
%     X_testOVA{class,1} = X_test;
%     Y_testOVA{class,1} = Y_test;
%     X_valOVA{class,1} = X_val;
%     Y_valOVA{class,1} = Y_val;
%     X_trainOVA{class,1} = X_train;
%     Y_trainOVA{class,1} = Y_train;
    
end

disp([newline 'All models have been ran' newline])

close all;
clearvars -except Mdls_OVA resultsClass labelClass scoreClass confClass X_testOVA Y_testOVA ogYtest
clc;

%% Put together all results for clear graphic interpretation

for class = 1:size(resultsClass,2)
    for model = 1:size(resultsClass{class},2)
        statsAll((class-1)*size(resultsClass{class},2)+model,1) = resultsClass{class}{model}.Accuracy;
        statsAll((class-1)*size(resultsClass{class},2)+model,2) = resultsClass{class}{model}.F1_score;
        statsAll((class-1)*size(resultsClass{class},2)+model,3) = resultsClass{class}{model}.MatthewsCorrelationCoefficient;
        
        stats{class}(model,1) = resultsClass{class}{model}.Accuracy;
        stats{class}(model,2) = resultsClass{class}{model}.F1_score;
        stats{class}(model,3) = resultsClass{class}{model}.MatthewsCorrelationCoefficient;
    end
end

%% Grab best model and predict on X test to see which class should be chosen

for class = 1:size(resultsClass,2)
    [tempRank,indRank] = sort(stats{class}(:,2));
    bestMdl{class,1} = Mdls_OVA{class}(indRank(end-14:end));
    bestMdl{class,2} = indRank(end-14:end);
end

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

% save('FinalOutputsOVA_Comb.mat','finalPredicts','C_test','resFinal','AUC','bestAUCs','bestMdl','stats','statsAll',...
%     'ogYtest','combScoresOVA','labelsOVA','scoresOVA','Cmats','resultsOVA','-v7.3')

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