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

leads = 16; K = 2;

P = nchoosek(1:leads,K);
% P=reshape(P(:,perms(1:K)),[],K);

seed1 = 13; % Seed for picking random features/lead combos
featuresTot = 29; % Total features
numRandPerms = 5;

rng(seed1);

randPermList = randperm(size(P,1));

P = P(randPermList,:);
load responseVars.mat

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
        
        tempData(:,end+1) = tempData(:,31);
        tempData(:,31) = respVars;
        
        mdlDataPermClass{perm,class} = tempData;
        featDataOVA{perm,class} = featNamesComOVA{perm};
        
        toc
    end
end

rng(0);

close all;
clearvars -except mdlDataPermClass featNamesComOVA featDataOVA;
clc;

%% KFold Adaption for per Subject

numSubs = 46;

for kfold = 1:numSubs
    
    disp([newline 'Running Models for K-Fold: ' num2str(kfold) newline])
    
    tClass = tic;
    
    for class = 1:5
       
%         mdlClassData = modelData{class};
        
        tic
        
        parfor model = 1:size(mdlDataPermClass,1)
            
            modelData = cell2mat(mdlDataPermClass{model,class});

            X_test = modelData(modelData(:,32) == kfold,1:30);
            Y_test = categorical(modelData(modelData(:,32) == kfold,31));
            trueLabelTest = modelData(modelData(:,32) == kfold,end);
            
            tempData = modelData(modelData(:,32) ~= kfold,1:31);
            
            [X,synthClass] = smote(tempData(:,1:30),[],'Class',tempData(:,31));
            
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

save('KFold_Mdl_Outputs_Comb.mat','labelsKFold','scoresKFold','CKFold','resultsKFold','testLabelsKFold','-v7.3');

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
    
    for class = 1:5
        
        tallyBest = zeros(1,600);
        
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
                bestFeats{mdls} = featDataOVA{crntMdl,class};
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
        scoresCentroidKFold{kfold} = scoresCentroid;
        [~,finalPredictsKFold{kfold}] = max(scoresCentroidKFold{kfold},[],2);
        finalConfKFold{kfold} = confusionmat(testLabelsBestMdl,finalPredictsKFold{kfold});
        finalResKFold{kfold} = getValues(finalConfKFold{kfold});
        
        finalStatsKFold(kfold,1) = finalResKFold{kfold}.Accuracy;
        finalStatsKFold(kfold,2) = finalResKFold{kfold}.F1_score;
        finalStatsKFold(kfold,3) = finalResKFold{kfold}.MatthewsCorrelationCoefficient;
        
        bestFeatsKFold{kfold} = bestFeatsClass;
        
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
    
    % saveas(gcf,'Top_15_Models_Comb.svg')
    % saveas(gcf,'Top_15_Models_Comb.png')
    
    % save('FinalOutputs_KFold_OVA_Comb_5.mat','scoresCentroidKFold','finalPredictsKFold','finalConfKFold','finalResKFold',...
    %     'finalStatsKFold','finalConfMat','finalResConfMat','bestMdlsKFold','bestFeatsClass','-v7.3');
    
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
title('Calculation of Final Results over Top Models for Double Lead')
xlabel('Number of Selected Top Models','FontWeight','bold')
% ylabel('AUC','FontWeight','bold')
legend({'AUC','F1','ACC','MCC'})

%%
titles_Plots = {'Task Precursor','Audio Task Interruption','Task Execution','Task Recovery','OTHER'};
% title_Features = {'Spectral Intensity','Temporal Cross Band Entropy','Engagement Index'};

figure;
t = tiledlayout(2,3);
for class = 1:size(tallyBestClass,1)
    
    x = 1:600;
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
        plot(x,ones(1,600)*top3(i),'-*','LineWidth',2,'Color',colors{i},'MarkerIndices',find(ones(1,600)*top3(i) == y))
    end
    
end
linkaxes([ax1,ax2,ax3,ax4,ax5],'xy')

title(t,'Top Models for Each Task State for Combined Lead Models')

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

for class = 1:length(bestFeatsKFold{1})
    countLead = zeros(1,length(eeg_leads_names));
    countFeatures = zeros(1,29);
    
    for kfold = 1:46
        
        for model = 1:length(bestFeatsKFold{kfold}{class})
            
            feats = bestFeatsKFold{kfold}{class}{model};
            
            numFeats1 = cell2mat(feats(2,1:15));
            numFeats2 = cell2mat(feats(2,16:end));
            
            firstFeat = feats{1,1};
            secondFeat = feats{1,16};
            
            firstLeadFeats = str2double(extractAfter(firstFeat,"Lead "));
            secondsLeadFeats = str2double(extractAfter(secondFeat,"Lead "));
            
            countLead(firstLeadFeats) = countLead(firstLeadFeats) + 1;
            countLead(secondsLeadFeats) = countLead(secondsLeadFeats) + 1;
            
            countFeatures(numFeats1) = countFeatures(numFeats1) + 1;
            countFeatures(numFeats2) = countFeatures(numFeats2) + 1;
            
        end
    end
    countLeadClass{class} = countLead;
    countFeatClass(class,:) = countFeatures;
    for region = 1:4
        countLeadRegion{class}(region) = sum(countLead(regions{region}));
    end
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
saveas(gcf,'Top_15_Models_Comb_LeadCount.svg')

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
saveas(gcf,'Top_15_Models_Comb_RegionCount.svg')

%% 

for i = 1:5
   
    plot(15:-1:1,statsBestClass{i}(:,2),'-o')
    hold on
    
end

legend({'Task 1','Task 2','Task 3','Task 4','Task 5'})
title('Elbow Graphs for Combined Leads')
xlabel('Top Models')
ylabel('F1 Scores')

saveas(gcf,'Elbow_Top15_CombLead.png')
