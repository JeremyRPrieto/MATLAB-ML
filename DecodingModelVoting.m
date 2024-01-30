%% DATASET INITIALIZATION IF NOT CREATED
close all;
clearvars -except Mdls;
clc;

load 5class_eeg.mat
load column_names.mat

addpath('NAPS Fusion')

tempLeadNum = 0;
resp = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Takes 5 Class Datasets and Splits into Lead Based Datasets %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic

for names = 1:length(column_names)
    
    % Grabs lead number from original dataset
    lead_num = str2double(extractAfter(column_names{names},'Lead '));
    
    % Grabs the response variables for all datapoints
    if isnan(lead_num)
        responseVar_H(:,resp) = [column_names{names};num2cell(H_eeg_5(:,names))];
        responseVar_NH(:,resp) = [column_names{names};num2cell(NH_eeg_5(:,names))];
        resp = resp + 1;
    end
    
    % Resets array counter based on new leads
    if lead_num ~= tempLeadNum
        count = 1;
        tempLeadNum = lead_num;
    end
    
    if count == 1
        % HYPOXIA
        eval(['Lead' num2str(lead_num) '_H = [column_names{names};num2cell(H_eeg_5(:,names))];']);
        
        % NONHYPOXIA
        eval(['Lead' num2str(lead_num) '_NH = [column_names{names};num2cell(NH_eeg_5(:,names))];']);
    else
        % HYPOXIA
        tempArray = [column_names{names};num2cell(H_eeg_5(:,names))];
        eval(['Lead' num2str(lead_num) '_H = [Lead' num2str(lead_num) '_H tempArray];']);
        
        % NONHYPOXIA
        tempArray = [column_names{names};num2cell(NH_eeg_5(:,names))];
        eval(['Lead' num2str(lead_num) '_NH = [Lead' num2str(lead_num) '_NH tempArray];']);
    end
    
    % Array counter
    count = count + 1;
    
end

% Checks if folders exist before making new folders

if ~exist('Leads_Hyp','dir'); mkdir Leads_Hyp; end
if ~exist('Leads_NonHyp','dir'); mkdir Leads_NonHyp; end
if ~exist('Leads','dir'); mkdir Leads; end

% Saves all lead datasets to folders within directory

for i = 1:16
    eval(['Lead' num2str(i) ' = [Lead' num2str(i) '_NH; Lead' num2str(i) '_H(2:end,:)];']);
end

if ~exist('Leads','dir') && ~exist('Leads_Hyp','dir') && ~exist('Leads_NonHyp','dir')
    
    for i = 1:16
        
        eval(['save(''Leads_Hyp\Lead' num2str(i) '_H'',''Lead' num2str(i) '_H'')']);
        eval(['save(''Leads_NonHyp\Lead' num2str(i) '_NH'',''Lead' num2str(i) '_NH'')']);
        eval(['save(''Leads\Lead' num2str(i) ''',''Lead' num2str(i) ''')']);
        
    end
    
end

% Saves response variables

% responseVarTot = [responseVar_NH;responseVar_H(2:end,:)];
% if ~exist('responseVars.mat','file')
save('responseVars','responseVar_H','responseVar_NH')
% end

toc

%% LOADING DATASETS IF ALREADY CREATED
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loads Datasets into Workspace they are not loaded prior %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Adds directories to path if they exist
if exist('Leads_Hyp','dir'); addpath Leads_Hyp; end
if exist('Leads_NonHyp','dir'); addpath Leads_NonHyp; end
if exist('Leads','dir'); addpath Leads; end


%% RANDOM SEED GENERATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sets seeds for random number generation for repeatability %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic

seed1 = 13; % Seed for picking random features
seed2 = 31; % Seed for picking random subject

leads = 16; % Total leads
featuresTot = 29; % Total features

% features1 = 12; % Total wavelet features
% features2 = 10; % Total entropy features
% features3 = 7; % Total EI features

% Set seed for features
rng(seed1);
modelData = cell(16*16,2);
featNames = cell(16*16,1);
combModelData = cell(16*16,1);

%Sets up data to be used within each model
for i = 0:15
    
    eval(['tempLead = Lead' num2str(i+1) ';']);
    
    for rnd = 1:16
        randFeats = randperm(featuresTot,15);
        modelData{rnd+(i*16),1} = tempLead(:,sort(randFeats));
        modelData{rnd+(i*16),2} = [responseVar_NH;responseVar_H(2:end,:)];
        featNames{rnd+(i*16)}(1,:) = modelData{rnd+(i*16),1}(1,:);
        featNames{rnd+(i*16)}(2,:) = num2cell(sort(randFeats));
    end
    
end

for i = 1:size(modelData,1)
    
    combModelData{i,1} = [modelData{i,1} modelData{i,2}];

end

save('ModelFeatures.mat','featNames','-v7.3')

% Reset RNG settings
rng(0);

toc

%% Model Creation from All Datasets
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Runs through all datasets created for 256 models of bagged forests %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Sets seed for random subject to be used as test data
rng(seed2);
testSubNum = randperm(46,1);

% Mdls = cell(size(modelData,1),1); 
X_test = cell(size(modelData,1),1); Y_test = cell(size(modelData,1),1);
X_train = cell(size(modelData,1),1); Y_train = cell(size(modelData,1),1); 
labels = cell(size(modelData,1),1); scores = cell(size(modelData,1),1);
% 
% testData = combModelData{1}(cell2mat(combModelData{1}(2:end,17)) == testSubNum,:);

for data = 1:size(modelData,1)
    
    tic
    
    tempData = cell2mat(combModelData{data}(2:end,:));

    %%% Test Data for 5 Class %%%
    X_test{data} = tempData(tempData(:,17) == testSubNum,1:15); Y_test{data} = tempData(tempData(:,17) == testSubNum,16);
    Y_test{data} = categorical(Y_test{data}, unique(Y_test{data}), {'B','S','M','A','OTH'});
    
    %%% Generate Training and Validation Data Sets
    
    tempData = tempData(randperm(size(tempData, 1)), :);
    totData = tempData(tempData(:,17) ~= testSubNum,1:15); respVars = tempData(tempData(:,17) ~= testSubNum,16:end);
    
    %%% Train data for 5 CLASS %%%
    X_train{data} = totData; Y_train{data} = respVars(:,1); Y_train{data} = categorical(Y_train{data}, unique(Y_train{data}), {'B','S','M','A','OTH'});

    Mdls{data} = TreeBagger(150,X_train{data},Y_train{data},'OOBPredictorImportance','On',...
        'PredictorSelection','curvature','Method','classification','MinLeafSize',2);
    
    [labels{data},scores{data}] = predict(Mdls{data},X_test{data});
%     labels{data} = categorical(labels{data}, unique(labels{data}), {'B','S','M','A','OTH'});
    
    C{data} = confusionmat(categorical(labels{data}),Y_test{data});
    
    [results{data},~] = getValues(C{data});
    
%     C = confusionmat(Y_test{data},

%     plot(oobError(Mdls{data}));
%     xlabel('Number of Grown Trees');
%     ylabel('Out-of-Bag Mean Squared Error');
    
    toc
    
end

%%% Save Models and Scores so to not have to run everything again
% if ~exist('ML_Outputs.mat','file') ~= 0 && ~exist('Train_Test_Data.mat','file') ~= 0
% save('models.mat','Mdls','-v7.3')
save('ML_Outputs.mat','scores','labels','results','C','-v7.3')
save('Train_Test_Data.mat','X_train','Y_train','X_test','Y_test','-v7.3')
% end

%% DISTANCE METRIC WITH SCORES

if ~exist('labels','var') ~= 0 && ~exist('X_train','var') ~= 0 && ~exist('featNames','var') ~= 0
    load ML_Outputs.mat
    load Train_Test_Data.mat
    load ModelFeatures.mat
end

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
    

% for i = 1:length(Mdls)
%     
%     tic
%     
%     stats(i,1) = multiClassAUC(scores{i},testResp{i});
%     
% %     [test_labels{i},test_scores{i}] = predict(Mdls{i},X_test{i});
%     C_test{i} = confusionmat(categorical(testResp{i}),Y_test{i});
% %     [results_test{i},~] = getValues(C_test{i});
%     
%     toc
%     
% end

for i = 1:256
    stats(i,2) = results{i}.Accuracy;
    stats(i,3) = results{i}.F1_score;
    stats(i,4) = results{i}.MatthewsCorrelationCoefficient;
end

% temp = modelData{1,2}(2:end,:);

% responses = cell2mat(temp(cell2mat(temp(:,2)) == testSubNum,1));

% for samp = 1:size(scores,1)
%     
%     tempScores = scores{samp};
% %     responses = Y_test{samp};
%     
%     stats(samp,1) = multiClassAUC(tempScores,responses);
%     
% end

% save('modelStats.mat','stats','-v7.3')

[sortStats,topModels] = sort(stats);
bestMdls = topModels(end-14:end,3);

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

scoresBest = scoresBest ./ 15;

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

% [~,finalPredict] = max(tally,[],2);

C_test = confusionmat(finalPredict,testResp{1});
resFinal = getValues(C_test);
confusionchart(C_test)

statsFinal(1,:) = resFinal.Accuracy;
statsFinal(2,:) = resFinal.F1_score;
statsFinal(3,:) = resFinal.MatthewsCorrelationCoefficient;

for i = 1:size(bestMdls,1)
    bestAUCs(i) = multiClassAUC(scores{bestMdls(i)},testResp{bestMdls(i)});
end

% save('FinalOut_Single.mat','C_test','resFinal','statsFinal','bestMdls','bestLbls','finalPredict','-v7.3');