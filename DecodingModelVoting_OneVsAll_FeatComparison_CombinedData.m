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
spectralFeats = 1:12;
tcbeFeats = 13:21;
eiFeats = 22:28;
SampEnFeat = 29;

featInds = {spectralFeats;tcbeFeats;eiFeats;SampEnFeat};

rng(seed1);
numRandPerms = 32;

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
        
        for feats = 1:4
            
            tempInd = featInds{feats};
            tempData = tempLead(2:end,tempInd);
            tempFeats = tempLead(1,tempInd);
            
            tempMdlData = [tempData responseClass(2:end,:) responseVars(2:end,1)];
            
            if lead == 0
                modelDataTotal{feats} = tempMdlData;
            else
                modelDataTotal{feats} = [modelDataTotal{feats};tempMdlData];
            end
            
            featNames{feats}(1,:) = tempFeats;
            featNames{feats}(2,:) = num2cell(tempInd);
            
%             modelData{lead+1,feats} = [tempData responseClass(2:end,:) responseVars(2:end,1)];
%             featNames{lead+1,feats}(1,:) = tempFeats;
%             featNames{lead+1,feats}(2,:) = num2cell(tempInd);

            
        end
        
%         for rnd = 1:numRandPerms
%             randFeats = randperm(featuresTot,15);
%             
%             tempData = tempLead(2:end,sort(randFeats));
%             tempFeats = tempLead(1,sort(randFeats));
%             
%             modelData{rnd+(lead*numRandPerms),1} = [tempData responseClass(2:end,:) responseVars(2:end,1)];
%             featNames{rnd+(lead*numRandPerms)}(1,:) = tempFeats;
%             featNames{rnd+(lead*numRandPerms)}(2,:) = num2cell(sort(randFeats));
%             
%         end
    end
    modelDataOVA{class} = modelDataTotal;
    featDataOVA{class} = featNames';
end


%% Creation of Split Model Data
% for class = 1:length(modelDataOVA)
%     for feats = 1:length(modelDataOVA{class})
%         
%         tempData = cell2mat(modelDataOVA{class}{feats});
%         
%         for rnd = 1:numRandPerms
%             randInds = randperm(size(tempData,1),size(tempData,1) / 16);
%             
%             randData = tempData(sort(randInds),:);
%             
%             modelDataOVA{class}{rnd,feats} = randData; 
%         end
%     end
% end

rng(0);

close all;
clearvars -except modelDataOVA featDataOVA;
clc;

%% KFold Adaption for per Subject

numSubs = 46;

for kfold = 1:numSubs
    
    disp([newline 'Running Models for K-Fold: ' num2str(kfold) newline])
    
    tClass = tic;
    
    for class = 1:5
       
        mdlClassData = modelDataOVA{class};
        featClassData = featDataOVA{class};
        
        for feats = 1:4
            
            mdlFeatData = mdlClassData(:,feats);
            featFeatData = featClassData(feats,:);
            
            tic
        
            parfor model = 1:size(mdlFeatData,1)
                
                modelData = cell2mat(mdlFeatData{model});
                featData = featFeatData{model};
                numFeats = size(featData,2);
                
                X_test = modelData(modelData(:,numFeats+2) == kfold,1:numFeats);
                Y_test = categorical(modelData(modelData(:,numFeats+2) == kfold,numFeats+1));
                trueLabelTest = modelData(modelData(:,numFeats+2) == kfold,end);
                
                tempData = modelData(modelData(:,numFeats+2) ~= kfold,1:numFeats+1);
                
                [X,synthClass] = smote(tempData(:,1:numFeats),[],'Class',tempData(:,numFeats+1));
                
                synthData = [X synthClass];
                
                synthData = synthData(randperm(size(synthData,1)),:);
                
                X_train = synthData(:,1:end-1); Y_train = categorical(synthData(:,end));
                
                Mdls = TreeBagger(150,X_train,Y_train,'PredictorSelection','curvature',...
                    'Method','classification','MinLeafSize',2);
                
                [labels{model},scores{model}] = predict(Mdls,X_test);
                
                C{model} = confusionmat(Y_test,categorical(labels{model}),'Order',{'1','0'});
                
                [results{model},~] = getValues(C{model});
                
                %             testData{model} = X_test;
                %             testDataY{model} = Y_test;
                testLabels{model} = trueLabelTest;
                
            end
            
            toc
        
            labelsFeats{feats} = labels;
            scoresFeats{feats} = scores;
            CFeats{feats} = C;
            resultsFeats{feats} = results;
            testLabelFeats{feats} = testLabels;
            
            clear labels scores C results testLabels
            
        end
        
        labelsClass{class} = labelsFeats;
        scoresClass{class} = scoresFeats;
        CClass{class} = CFeats;
        resultsClass{class} = resultsFeats;
%         testDataClass{class} = testData;
%         testDataYClass{class} = testDataY;
        testLabelsClass{class} = testLabelFeats;
        
        clear labelsFeats scoresFeats CFeats resultsFeats testLabelFeats
        
    end
    
    tEnd = toc(tClass);
    
    disp([newline 'Program Time for K-Fold: ' num2str(tEnd) ' seconds'])
    
    labelsKFold{kfold} = labelsClass;
    scoresKFold{kfold} = scoresClass;
    CKFold{kfold} = CClass;
    resultsKFold{kfold} = resultsClass;
%     testDataKFold{kfold} = testDataClass;
%     testDataYKFold{kfold} = testDataYClass;
    testLabelsKFold{kfold} = testLabelsClass;
    
    clear labelsClass scoresClass CClass resultsClass testLabelsClass
    
end

disp([newline 'All models have been ran' newline])

save('FeatComp_Combined_Mdl_Outputs.mat','featDataOVA','labelsKFold','scoresKFold','CKFold','resultsKFold','testLabelsKFold','-v7.3');

%% KFolds and Results of Each Feature Set
for kfold = 1:length(resultsKFold)
    dataKFold = resultsKFold{kfold};
    for class = 1:length(dataKFold)
        dataClass = dataKFold{class};
        for feats = 1:length(dataClass)
            dataFeats = dataClass{feats};
            for model = 1:length(dataFeats)
                dataMdl = dataFeats{model};
                
                statsMdl(model,1) = dataMdl.Accuracy;
                statsMdl(model,2) = dataMdl.F1_score;
                statsMdl(model,3) = dataMdl.MatthewsCorrelationCoefficient;
            end
            statsFeats{feats} = statsMdl;
        end
        statsClass{class} = statsFeats;
    end
    statsKFold{kfold} = statsClass;
end
% 
% for kfold = 1:length(statsKFold)
%     resKFold = statsKFold{kfold};
%     for class = 1:length(resKFold)
%         resClass = resKFold{class};
%         for feats = 1:length(resClass)
%             resFeats = resClass{feats};
%             
%             resMdl = resFeats(:,2);
%             [tempRank,indRank] = sort(resMdl);
%             tempInd = ~isnan(tempRank);
%             ranks = indRank(tempInd);
%             
%             bestMdlsFeats{feats} = ranks(end);
%         end
%         bestMdlsClass{class} = bestMdlsFeats;
%     end
%     bestMdlsKFold{kfold} = bestMdlsClass;
% end

for kfold = 1:length(statsKFold)
    for class = 1:length(statsKFold{kfold})
        for feats = 1:length(statsKFold{kfold}{class})
%             bestMdls = bestMdlsKFold{kfold}{class}{feats};
            
            bestFeats = featDataOVA{class}{feats};
            labelsBestMdl = labelsKFold{kfold}{class}{feats}{1};
            scoresBestMdl = scoresKFold{kfold}{class}{feats}{1}(:,2);
            confMatBestMdl = CKFold{kfold}{class}{feats}{1};
            testLabelsBestMdl = testLabelsKFold{kfold}{class}{feats}{1};
            
            scoresFeats(:,feats) = scoresBestMdl;
            
        end
        scoresClass{class} = scoresFeats;
        
        clear scoresFeats
    end
    scoresKFold{kfold} = scoresClass;
    testLabels{kfold} = testLabelsBestMdl;
    
    clear scoresClass
end

%% New Organization of Scores
for kfold = 1:length(scoresKFold)
    for feats = 1:4
        for class = 1:5
            
            tempScore = scoresKFold{kfold}{class}(:,feats);
            
            scoresMat(:,class) = tempScore;
            
        end
        scoresFeats{feats} = scoresMat;
        
        scoresKFold_ReOrg{kfold,feats} = scoresMat;
        clear scoresMat
    end
%     scoresKFold_ReOrg{kfold} = scoresFeats;
end

for kfold = 1:size(scoresKFold_ReOrg,1)
    for feats = 1:size(scoresKFold_ReOrg,2)
        
        [~,finalPredictsKFold{kfold,feats}] = max(scoresKFold_ReOrg{kfold,feats},[],2);
        finalConfKFold{kfold,feats} = confusionmat(testLabels{kfold},finalPredictsKFold{kfold,feats});
        finalResKFold{kfold,feats} = getValues(finalConfKFold{kfold,feats});
        
        finalStatsKFold{kfold,feats}(:,1) = finalResKFold{kfold,feats}.Accuracy;
        finalStatsKFold{kfold,feats}(:,2) = finalResKFold{kfold,feats}.F1_score;
        finalStatsKFold{kfold,feats}(:,3) = finalResKFold{kfold,feats}.MatthewsCorrelationCoefficient;
        
    end
end

%% Final Confusion Matrices per Feature
finalConfMat_Feats = cell(1,4);
finalResConfMat_Feats = cell(1,4);

for feats = 1:size(finalConfKFold,2)
    finalConfMat = zeros(5,5);
    for kfold = 1:size(finalConfKFold,1)
        finalConfMat = finalConfMat + finalConfKFold{kfold,feats};
    end
    finalConfMat_Feats{feats} = finalConfMat;
    finalResConfMat_Feats{feats} = getValues(finalConfMat_Feats{feats});
    finalStatsConfMat{feats}(:,1) = finalResConfMat_Feats{feats}.Accuracy;
    finalStatsConfMat{feats}(:,2) = finalResConfMat_Feats{feats}.F1_score;
    finalStatsConfMat{feats}(:,3) = finalResConfMat_Feats{feats}.MatthewsCorrelationCoefficient;
end

for res = 1:4
    [~,finalStatsConfMat{res}(:,4)] = AUC_Calc_MultiClass(finalConfMat_Feats{res});
end

% save('FeatComp_Combined_FinalOutputs.mat','finalConfMat_Feats','finalResConfMat_Feats','finalStatsConfMat','scoresKFold_ReOrg','finalPredictsKFold',...
%     'finalConfKFold','finalResKFold','finalStatsKFold','-v7.3')