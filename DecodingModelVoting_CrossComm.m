close all;
clearvars;
clc;

%% ALL POSSIBLE PERMS OF CROSS TALK

% Adds directories to path if they exist
if exist('Leads_Hyp','dir'); addpath Leads_Hyp; end
if exist('Leads_NonHyp','dir'); addpath Leads_NonHyp; end
if exist('Leads','dir'); addpath Leads; end

leads = 16; K = 2;

P = nchoosek(1:leads,K);
P=reshape(P(:,perms(1:K)),[],K);

folderInfo = dir('Leads');

for files = 3:length(folderInfo)
    filename = folderInfo(files).name;
    load(filename)
end

load responseVars

seed1 = 13; % Seed for picking random features/lead combos
featuresTot = 29; % Total features

rng(seed1);

randPermList = randperm(size(P,1));

P = P(randPermList,:);

tic

for perm = 1:size(P,1)
    
    lead1 = P(perm,1); lead2 = P(perm,2);
    eval(['tempLead1 = Lead' num2str(lead1) ';']);
    eval(['tempLead2 = Lead' num2str(lead2) ';']);
    
    randFeats = randperm(featuresTot,15);
    responseVars = [responseVar_NH;responseVar_H(2:end,:)];
    
    modelDataCom{perm,1} = [tempLead1(:,sort(randFeats)) tempLead2(:,sort(randFeats)) responseVars];
    featNamesCom{perm}(1,:) = modelDataCom{perm,1}(1,1:15*2);
    featNamesCom{perm}(2,:) = num2cell([sort(randFeats) sort(randFeats)]);
    
end

toc

% tic
% save('LeadComboModelFeats.Mat','featNamesCom','modelDataCom','-v7.3');
% toc

rng(0);

%% MODEL CREATION
close all;
clearvars -except modelDataCom Mdls_Com;
clc;

seed2 = 31; % Seed for picking random subject
rng(seed2);
testSubNum = randperm(46,1);

% Mdls_Com = cell(size(modelDataCom,1),1);
X_test = cell(size(modelDataCom,1),1); Y_test = cell(size(modelDataCom,1),1);
X_train = cell(size(modelDataCom,1),1); Y_train = cell(size(modelDataCom,1),1); 
labels = cell(size(modelDataCom,1),1); scores = cell(size(modelDataCom,1),1);

for data = 1:size(modelDataCom,1)
    
    tic
    
    tempData = cell2mat(modelDataCom{data}(2:end,:));

    %%% Test Data for 5 Class %%%
    X_test{data} = tempData(tempData(:,32) == testSubNum,1:30); Y_test{data} = tempData(tempData(:,32) == testSubNum,31);
    Y_test{data} = categorical(Y_test{data}, unique(Y_test{data}), {'B','S','M','A','OTH'});
    
    %%% Generate Training and Validation Data Sets
    
    tempData = tempData(randperm(size(tempData, 1)), :);
    totData = tempData(tempData(:,32) ~= testSubNum,1:30); respVars = tempData(tempData(:,32) ~= testSubNum,31:end);
    
    %%% Train data for 5 CLASS %%%
    X_train{data} = totData; Y_train{data} = respVars(:,1); Y_train{data} = categorical(Y_train{data}, unique(Y_train{data}), {'B','S','M','A','OTH'});

%     Mdls_Com{data} = TreeBagger(150,X_train{data},Y_train{data},'OOBPredictorImportance','On',...
%         'PredictorSelection','curvature','Method','classification','MinLeafSize',2,'ClassNames',{'B','S','M','A','OTH'});
    
    [labels{data},scores{data}] = predict(Mdls_Com{data},X_test{data});
%     labels{data} = categorical(labels{data}, unique(labels{data}), {'B','S','M','A','OTH'});
    
    C{data} = confusionmat(categorical(labels{data},{'B','S','M','A','OTH'}),Y_test{data});
    
    [results{data},~] = getValues(C{data});

    toc
    
end

%% GRAB STATS FOR VOTING
for i = 1:length(Y_test)
    
    tempData = Y_test{i};
   
    for j = 1:size(Y_test{i},1)
        
        tempData = string(tempData);
        
        if strcmp(tempData(j),'B')
            testResp{i}(j,1) = 1;
        elseif strcmp(tempData(j),'S')
            testResp{i}(j,1) = 2;
        elseif strcmp(tempData(j),'M')
            testResp{i}(j,1) = 3;
        elseif strcmp(tempData(j),'A')
            testResp{i}(j,1) = 4;
        else
            testResp{i}(j,1) = 5;
        end
        
    end
    
end

for i = 1:length(scores)
    stats(i,1) = results{i}.Accuracy;
    stats(i,2) = results{i}.F1_score;
    stats(i,3) = results{i}.MatthewsCorrelationCoefficient;
    stats(i,4) = multiClassAUC(scores{i},testResp{i});
end

[sortStats,topModels] = sort(stats);
bestMdls = topModels(end-12:end,2);

for i = 1:size(bestMdls,1)
    
    bestLbls(:,i) = labels{bestMdls(i)};
    
end

classLabels = {'B','S','M','A','OTH'};

for i = 1:size(bestLbls,1)
    
    for classes = 1:5
        
        tally(i,classes) = sum(strcmp(bestLbls(i,:),classLabels{classes}));
        
    end
    
end

scoresBest = 0;

for i = 1:size(bestMdls,1)
    scoresBest = scoresBest + scores{bestMdls(i)};
%     certaintyMat{i} = CentroidMethodGen(scores{bestMdls(i)});
end

scoresBest = scoresBest ./ 17;

for i = 1:size(tally,1)
    tempRow = tally(i,:);
    tempScores = scoresBest(i,:);
    val = max(tempRow);
    
    locVals = find(tempRow == val);
    
    if length(locVals) == 1
        finalPredict(i,1) = locVals;
    else 
        [~,maxInd] = max(tempScores(locVals));
        finalPredict(i,1) = locVals(maxInd);
    end
end
% 
% [~,finalPredict] = max(tally,[],2);

C_test = confusionmat(finalPredict,testResp{1});
resFinal = getValues(C_test)

for i = 1:size(bestMdls,1)
    bestAUCs(i) = multiClassAUC(scores{bestMdls(i)},testResp{1});
end

% save('FinalOut_Comb.mat','bestLbls','bestMdls','C','C_test','finalPredict','scores','stats','resFinal')
