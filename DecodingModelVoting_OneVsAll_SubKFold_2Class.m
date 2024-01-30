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

for class = 0:1
    for lead = 0:15
        
        eval(['tempLead = Lead' num2str(lead+1) ';']);
        responseVars = [responseVar_NH;responseVar_H(2:end,:)];
        responseClass = responseVars;
        
        classVars = cell2mat(responseVars(2:end,3)) == class;
        
        for vars = 1:size(responseVars,1)-1
            if ~classVars(vars)
                responseClass{vars+1,3} = 0;
            else
                responseClass{vars+1,3} = 1;
            end
        end
        
        for rnd = 1:numRandPerms
            randFeats = randperm(featuresTot,15);
            
            tempData = tempLead(2:end,sort(randFeats));
            tempFeats = tempLead(1,sort(randFeats));
            
            modelData{rnd+(lead*numRandPerms),1} = [tempData responseClass(2:end,:) responseVars(2:end,3)];
            featNames{rnd+(lead*numRandPerms)}(1,:) = tempFeats;
            featNames{rnd+(lead*numRandPerms)}(2,:) = num2cell(sort(randFeats));
            
        end
    end
    modelDataOVA{class+1} = modelData;
    featDataOVA{class+1} = featNames';
end

rng(0);

close all;
clearvars -except modelDataOVA featDataOVA ogTestData;
clc;

%% KFold Adaption for per Subject

numSubs = 46;

for kfold = 1:numSubs
    
    disp([newline 'Running Models for K-Fold: ' num2str(kfold) newline])
    
    tClass = tic;
    
    for class = 1:2
       
        mdlClassData = modelDataOVA{class};
        
        tic
        
        parfor model = 1:length(mdlClassData)
            
            modelData = cell2mat(mdlClassData{model});

            X_test = modelData(modelData(:,17) == kfold,1:15);
            Y_test = categorical(modelData(modelData(:,17) == kfold,18));
            trueLabelTest = modelData(modelData(:,17) == kfold,end);
            
            tempData = modelData(modelData(:,17) ~= kfold,[1:15,18]);
            
%             [X,synthClass] = smote(tempData(:,1:15),[],'Class',tempData(:,16));
%             
%             synthData = [X synthClass];
%             
%             synthData = synthData(randperm(size(synthData,1)),:);
%             
            X_train = tempData(:,1:end-1); Y_train = categorical(tempData(:,end));
            
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
        
        labelsClass{class} = labels;
        scoresClass{class} = scores;
        CClass{class} = C;
        resultsClass{class} = results;
%         testDataClass{class} = testData;
%         testDataYClass{class} = testDataY;
        testLabelsClass{class} = testLabels;
        
        clear labels scores C results testLabels
        
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

save('KFold_Mdl_Outputs_2Class.mat','featDataOVA','labelsKFold','scoresKFold','CKFold','resultsKFold','testLabelsKFold','-v7.3');

%% Combination of all KFolds for One vs All Combination

for kfold = 1:length(resultsKFold)
    dataKFold = resultsKFold{kfold};
    for class = 1:length(dataKFold)
        dataClass = dataKFold{class}; 
        for model = 1:length(dataClass) 
            dataMdl = dataClass{model};
            statsMdl(model,1) = dataMdl.Accuracy;
            statsMdl(model,2) = dataMdl.F1_score;
            statsMdl(model,3) = dataMdl.MatthewsCorrelationCoefficient;   
        end
        statsClass{class} = statsMdl;  
    end
    statsKFold{kfold} = statsClass;
end

for kfold = 1:length(statsKFold)
    resKFold = statsKFold{kfold};
    for class = 1:length(resKFold)
        resClass = resKFold{class};
        
        resMdl = resClass(:,2);
        [tempRank,indRank] = sort(resMdl);
        tempInd = ~isnan(tempRank);
        ranks = indRank(tempInd);
        
        bestMdlsClass{class} = ranks(end-14:end);
    end
    bestMdlsKFold{kfold} = bestMdlsClass;
end

for kfold = 1:length(bestMdlsKFold)
    for class = 1:length(bestMdlsKFold{kfold})
        bestMdls = bestMdlsKFold{kfold}{class};
        
        for mdls = 1:length(bestMdls)
            crntMdl = bestMdls(mdls);
%             bestFeats{mdls} = featDataOVA{class}{crntMdl};
            labelsBestMdl = labelsKFold{kfold}{class}{crntMdl};
            scoresBestMdl = scoresKFold{kfold}{class}{crntMdl};
            confMatBestMdl = CKFold{kfold}{class}{crntMdl};
            testLabelsBestMdl = testLabelsKFold{kfold}{class}{crntMdl};
%             testYLabelsBestMdl = testDataYKFold{kfold}{class}{crntMdl};
            
            if mdls == 1
                combScoresBestMdl = scoresBestMdl(:,2);
            else
                combScoresBestMdl = [combScoresBestMdl scoresBestMdl(:,2)];
            end
        end
        scoresCentroid(:,class) = CentroidMethodGen(combScoresBestMdl);
%         bestFeatsClass{class} = bestFeats;
        
        clear combScoresBestMdl bestFeats
    end
    scoresCentroidKFold{kfold} = scoresCentroid;
    [~,finalPredictsKFold{kfold}] = max(scoresCentroidKFold{kfold},[],2);
    
    indTrue = finalPredictsKFold{kfold} == 2;
    finalPredictsKFold{kfold}(~indTrue) = 0; finalPredictsKFold{kfold}(indTrue) = 1;
    
    finalConfKFold{kfold} = confusionmat(testLabelsBestMdl,finalPredictsKFold{kfold});
    finalResKFold{kfold} = getValues(finalConfKFold{kfold});
    
    finalStatsKFold(kfold,1) = finalResKFold{kfold}.Accuracy;
    finalStatsKFold(kfold,2) = finalResKFold{kfold}.F1_score;
    finalStatsKFold(kfold,3) = finalResKFold{kfold}.MatthewsCorrelationCoefficient;
    
    clear scoresCentroid
end

finalConfMat = zeros(size(finalConfKFold{1}));

for kfold = 1:length(finalConfKFold)
    finalConfMat = finalConfMat + finalConfKFold{kfold};
end

[finalResConfMat,~] = getValues(finalConfMat);

[~,avgAUC] = AUC_Calc_MultiClass(finalConfMat,2);

% AUC_Calc

cm = confusionchart(finalConfMat,{'NH','H'},'Title',{join(["AUC:"; string(round(avgAUC,3)); "ACC:"; string(round(finalResConfMat.Accuracy,3));...
    "F1:"; string(round(finalResConfMat.F1_score,3)); "MCC:"; string(round(finalResConfMat.MatthewsCorrelationCoefficient,3))])});
sortClasses(cm,{'NH','H'})

saveas(gcf,'Top_15_Models_2Class.svg')

% save('FinalOutputs_KFold_OVA_15.mat','scoresCentroidKFold','finalPredictsKFold','finalConfKFold','finalResKFold',...
%     'finalStatsKFold','finalConfMat','finalResConfMat','bestMdlsKFold','bestFeatsClass','-v7.3');

%% Lead Extraction for best lead for classification

eeg_leads_names = {'P7', 'Cz', 'POz', 'Pz', 'P8', 'O1', 'Oz', 'O2', 'AF3', 'AF4', 'F1', 'F2', 'C1', 'C2', 'P3', 'P4'};

countLeadClass = cell(1,5);
countLeadRegion = cell(1,5);

parietal = [1,3,4,5,15,16];
central = [2,13,14];
frontal = 9:12;
occipital = 6:8;

regions = {parietal,central,frontal,occipital};

for class = 1:length(bestFeatsClass)
    countLead = zeros(1,length(eeg_leads_names));
    for model = 1:length(bestFeatsClass{class})
    
        feats = bestFeatsClass{class}{model};
        
        singleFeat = feats{1};
        
        leadFeats = str2double(extractAfter(singleFeat,"Lead "));
        
        countLead(leadFeats) = countLead(leadFeats) + 1;
        
    end
    countLeadClass{class} = countLead;
    for region = 1:4
        countLeadRegion{class}(region) = sum(countLead(regions{region}));
    end
end

figure;

t = tiledlayout(2,3);

titles_Plots = {'Task Precursor','Audio Task Interruption','Task Execution','Task Recovery','OTHER'};

for class = 1:length(countLeadClass)

    x = categorical(eeg_leads_names);
    x = reordercats(x,eeg_leads_names);
    y = countLeadClass{class};
    
%     ticks = 0:1:max(y);
    
    eval(['ax' num2str(class) ' = nexttile;']);
    bar(x,y)
    title(titles_Plots{class})
    xlabel('Leads')
    ylabel('Count')
%     yticks(ticks)
    
    set(gca,'fontweight','bold')
    
end

linkaxes([ax1,ax2,ax3,ax4,ax5],'xy')

title(t,'Count of Top Leads per each Task State')

set(gcf, 'Position', get(0, 'Screensize'));
saveas(gcf,'Top_15_Models_LeadCount.svg')

% Figure for region count
figure;

t = tiledlayout(2,3);

titles_Plots = {'Task Precursor','Audio Task Interruption','Task Execution','Task Recovery','OTHER'};

regionNames = {'Parietal','Central','Frontal','Occipital'};
regionShort = {'P','C','F','O'};

for class = 1:length(countLeadRegion)

    x = categorical(regionNames);
    x = reordercats(x,regionNames);
    y = countLeadRegion{class};
    
%     ticks = 0:1:max(y);
    
    eval(['ax' num2str(class) ' = nexttile;']);
    bar(x,y)
    title(titles_Plots{class})
    xlabel('Regions')
    ylabel('Count')
%     yticks(ticks)
    
    set(gca,'fontweight','bold')
    
end

linkaxes([ax1,ax2,ax3,ax4,ax5],'xy')

title(t,'Count of Top Regions per each Task State')

set(gcf, 'Position', get(0, 'Screensize'));
saveas(gcf,'Top_15_Models_RegionCount.svg')
