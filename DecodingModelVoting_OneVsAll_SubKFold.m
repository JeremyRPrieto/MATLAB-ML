close all;
clearvars;
clc;

addpath('C:\Users\jeremyprieto31\OneDrive\Documents\Graduate\MATLAB Toolboxes')

% addpath('C:\Users\jerem\OneDrive\Documents\Graduate\MATLAB Toolboxes')

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

save('KFold_Mdl_Outputs.mat','labelsKFold','scoresKFold','CKFold','resultsKFold','testLabelsKFold','-v7.3');

%% Combination of all KFolds for One vs All Combination

for topNum = 3
    
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
            
            bestMdlsClass{class} = ranks(end-topNum:end);
        end
        bestMdlsKFold{kfold} = bestMdlsClass;
    end
    
    colors = {[0 0.4470 0.7410],[0.8500 0.3250 0.0980],[0.9290 0.6940 0.1250],[0.4940 0.1840 0.5560],[0.4660 0.6740 0.1880]};
    
%     for kfold = 1:46
%         indF1 = zeros(15,5);
%         for class = 1:5
%     
%             mdls = bestMdlsKFold{kfold}{class};
%     
%             f1Scores = statsKFold{kfold}{class}(mdls,2);
%     
%             indF1(:,class) = f1Scores > 0.5;
%     
%             plot(15:-1:1,f1Scores,'-o','Color',colors{class});
%             hold on
%             stem(15:-1:1,indF1(:,class).*f1Scores,'filled','Color',colors{class},'HandleVisibility','off');
%     
%         end
%         hold off
%         
%         ylabel('F1 Scores of Top 15','FontWeight','bold')
%         xlabel('Model Number','FontWeight','bold')
%         legend({'T_P','T_A','T_E','T_R','OTH'})
%         title('Elbow Graph for Single Lead Models')
%     
%         [val_min,loc_min] = sort(sum(indF1));
%     
%         val_min = val_min(val_min~=0);
%     
%         rank_tops{kfold} = val_min(1);
%     
%     end
    
    % for kfold = 1:length(bestMdlsKFold)
    %     for class = 1:length(bestMdlsKFold{kfold})
    %
    %         ranks = bestMdlsKFold{kfold}{class};
    %         optiTops = rank_tops{kfold};
    %
    %         bestMdlsClass{class} = ranks(end-optiTops+1:end);
    %     end
    %     bestMdlsKFold{kfold} = bestMdlsClass;
    % end
    
    for class = 1:5
        
        tallyBest = zeros(1,256);
        
        for kfold = 1:46
            
            bestMdls = bestMdlsKFold{kfold}{class};
            
            for mdls = 1:length(bestMdls)
                crntMdl = bestMdls(mdls);
                tallyBest(crntMdl) = tallyBest(crntMdl) + 1;
            end
        end
        tallyBestClass(class,:) = tallyBest;
    end
    
    for kfold = 1:length(bestMdlsKFold)
        for class = 1:length(bestMdlsKFold{kfold})
            bestMdls = bestMdlsKFold{kfold}{class};
            
            for mdls = 1:length(bestMdls)
                crntMdl = bestMdls(mdls);
                bestFeats{mdls} = featDataOVA{class}{crntMdl};
                labelsBestMdl = labelsKFold{kfold}{class}{crntMdl};
                scoresBestMdl = scoresKFold{kfold}{class}{crntMdl};
                confMatBestMdl = CKFold{kfold}{class}{crntMdl};
                testLabelsBestMdl = testLabelsKFold{kfold}{class}{crntMdl};
                statsBestMdl = statsKFold{kfold}{class}(crntMdl,:);
                %             testYLabelsBestMdl = testDataYKFold{kfold}{class}{crntMdl};
                
                if mdls == 1
                    combScoresBestMdl = scoresBestMdl(:,2);
                    combStatsBestMdl = statsBestMdl;
                else
                    combScoresBestMdl = [combScoresBestMdl scoresBestMdl(:,2)];
                    combStatsBestMdl = [combStatsBestMdl;statsBestMdl];
                end
            end
            scoresCentroid(:,class) = CentroidMethodGen(combScoresBestMdl);
            bestFeatsClass{class} = bestFeats;
            statsBestClass{class} = combStatsBestMdl;
            
            clear combScoresBestMdl bestFeats combStatsBestMdl
        end
        bestFeatsKFold{kfold} = bestFeatsClass;
        scoresCentroidKFold{kfold} = scoresCentroid;
        [~,finalPredictsKFold{kfold}] = max(scoresCentroidKFold{kfold},[],2);
        finalConfKFold{kfold} = confusionmat(testLabelsBestMdl,finalPredictsKFold{kfold});
        finalResKFold{kfold} = getValues(finalConfKFold{kfold});
        
        finalStatsKFold(kfold,1) = finalResKFold{kfold}.Accuracy;
        finalStatsKFold(kfold,2) = finalResKFold{kfold}.F1_score;
        finalStatsKFold(kfold,3) = finalResKFold{kfold}.MatthewsCorrelationCoefficient;
        
        clear scoresCentroid bestFeatsClass
    end
    
    finalConfMat = zeros(5,5);
    
    for kfold = 1:length(finalConfKFold)
        finalConfMat = finalConfMat + finalConfKFold{kfold};
    end
    
    [finalResConfMat,~] = getValues(finalConfMat);
    
    AUC_Calc
    
    cm = confusionchart(finalConfMat,{'TP','TA','TE','TR','OTH'},'Title',{join(["AUC:"; string(round(avgAUC,3)); "ACC:"; string(round(finalResConfMat.Accuracy,3));...
        "F1:"; string(round(finalResConfMat.F1_score,3)); "MCC:"; string(round(finalResConfMat.MatthewsCorrelationCoefficient,3))])});
    sortClasses(cm,{'TP','TA','TE','TR','OTH'})
    
    mdlAUCs(topNum) = avgAUC;
    mdlF1(topNum) = finalResConfMat.F1_score;
    mdlACC(topNum) = finalResConfMat.Accuracy;
    mdlMCC(topNum) = finalResConfMat.MatthewsCorrelationCoefficient;
    
end

figure;
plot(mdlAUCs,'-o')
hold on
plot(mdlF1,'-o')
plot(mdlACC,'-o')
plot(mdlMCC,'-o')
xline(3,'LineWidth',2)
hold off
ylim([0 1])
xlim([1 topNum])
title('Calculation of Final Results over Top Models for Single Lead')
xlabel('Number of Selected Top Models','FontWeight','bold')
% ylabel('AUC','FontWeight','bold')
legend({'AUC','F1','ACC','MCC'})

% saveas(gcf,'Top_15_Models.svg')
% saveas(gcf,'Top_15_Models.png')

% save('FinalOutputs_KFold_OVA_15.mat','scoresCentroidKFold','finalPredictsKFold','finalConfKFold','finalResKFold',...
%     'finalStatsKFold','finalConfMat','finalResConfMat','bestMdlsKFold','bestFeatsClass','-v7.3');

%%
titles_Plots = {'Task Precursor','Audio Task Interruption','Task Execution','Task Recovery','OTHER'};
% title_Features = {'Spectral Intensity','Temporal Cross Band Entropy','Engagement Index'};

figure;
t = tiledlayout(2,3);
for class = 1:size(tallyBestClass,1)
    
    x = 1:256;
    y = tallyBestClass(class,:);
    
    %     ticks = 0:1:max(y);
    
    eval(['ax' num2str(class) ' = nexttile;']);
    bar(x,y)
    title(titles_Plots{class})
    xlabel('Feature Models')
    ylabel('Number of Appearances in Top 4')
    %     yticks(ticks)
    
    set(gca,'fontweight','bold')
    
    hold on
    
    vals = unique(y);
    top3 = vals(end-2:end);
    
    colors = {[0.4940 0.1840 0.5560],[0.8500 0.3250 0.0980],[0.4660 0.6740 0.1880]};
    
    for i = 1:length(top3)
        plot(x,ones(1,256)*top3(i),'-*','LineWidth',2,'Color',colors{i},'MarkerIndices',find(ones(1,256)*top3(i) == y))
    end
    
end
linkaxes([ax1,ax2,ax3,ax4,ax5],'xy')

title(t,'Top Models for Each Task State for Single Lead Models')

set(gcf, 'Position', get(0, 'Screensize'));

%% Lead Extraction for best lead for classification

eeg_leads_names = {'P7', 'Cz', 'POz', 'Pz', 'P8', 'O1', 'Oz', 'O2', 'AF3', 'AF4', 'F1', 'F2', 'C1', 'C2', 'P3', 'P4'};

countLeadClass = cell(1,5);
countLeadRegion = cell(1,5);

parietal = [1,3,4,5,15,16];
central = [2,13,14];
frontal = 9:12;
occipital = 6:8;

regions = {parietal,central,frontal,occipital};

spectralFeats = 1:12;
tcbeFeats = 13:21;
eiFeats = 22:28;
SampEnFeat = 29;

featInds = {spectralFeats;tcbeFeats;eiFeats;SampEnFeat};

for class = 1:length(bestFeatsKFold{1})
    countLead = zeros(1,length(eeg_leads_names));
    countFeatures = zeros(1,29);
    
    for kfold = 1:46
    
        for model = 1:length(bestFeatsKFold{kfold}{class})
            
            feats = bestFeatsKFold{kfold}{class}{model};
            numFeats = cell2mat(feats(2,:));
            
            singleFeat = feats{1};
            
            leadFeats = str2double(extractAfter(singleFeat,"Lead "));
            
            countLead(leadFeats) = countLead(leadFeats) + 1;
            countFeatures(numFeats) = countFeatures(numFeats) + 1;
        end
        
    end
    countLeadClass{class} = countLead;
    countFeatClass(class,:) = countFeatures;
    for region = 1:4
        countLeadRegion{class}(region) = sum(countLead(regions{region}));
    end
    
    for feats = 1:3
        countFeatTotal{1,class}(feats) = sum(countFeatures(featInds{feats}));
    end
    countFeatTotal{2,class} = sum(countFeatTotal{1,class});
end

figure;

t = tiledlayout(2,3);

titles_Plots = {'Task Precursor','Audio Task Interruption','Task Execution','Task Recovery','OTHER'};
predictors_names = {'W_1', 'W_2', 'W_3' 'W_4' 'W_5', 'W_6', 'W_7', 'W_8' 'W_9' 'W_{10}', 'W_{11}' 'W_{12}' ...
    'ROC_F', 'ROC_{FD}' 'ROC_{TBA}', 'ROC_{BA}', 'ROC_B' 'ROC_{BE}' ...
    'ROC_{ABT}' 'ROC_{ABTE}' 'ROC_{GBA}' 'EI_S' 'EI_M' 'EI_{SD}' 'EIE_S' 'EIE_M' 'EIE_{SD}' 'EI' 'SE' };

for class = 1:size(countFeatClass,1)

    x = categorical(predictors_names);
    x = reordercats(x,predictors_names);
    y = countFeatClass(class,:);
    
%     ticks = 0:1:max(y);
    
    eval(['ax' num2str(class) ' = nexttile;']);
    bar(x,y)
    title(titles_Plots{class})
    xlabel('Features')
    ylabel('Count')
%     xtickangle(45)
%     yticks(ticks)
    
    set(gca,'fontweight','bold')
    
end

linkaxes([ax1,ax2,ax3,ax4,ax5],'xy')

title(t,'Count of Top Features per each Task State')

set(gcf, 'Position', get(0, 'Screensize'));

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
saveas(gcf,'Top_15_Models_LeadCount.png')
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
saveas(gcf,'Top_15_Models_RegionCount.png')
%% 

for i = 1:5
   
    plot(15:-1:1,statsBestClass{i}(:,2),'-o')
    hold on
    
end

legend({'Task 1','Task 2','Task 3','Task 4','Task 5'})
title('Elbow Graphs for Single Lead')
xlabel('Top Models')
ylabel('F1 Scores')

saveas(gcf,'Elbow_Top15_SingleLead.png')