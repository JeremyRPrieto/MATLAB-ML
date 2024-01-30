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

P = nchoosek(featInds{1},5);

rng(seed1);
numRandPerms = 16;
numRandFeats = 5;

for class = 1:5
    for lead = 0:15
        
        tic
        
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
        
        for tcbe = 1:length(featInds{2})
            tcbeInd = featInds{2}(tcbe);
            for perm = 1:numRandPerms
                rnd = randperm(size(P,1),1);
                specInd = featInds{1}(P(rnd,:));
                
                tempData1 = tempLead(2:end,tcbeInd);
                tempData2 = tempLead(2:end,specInd);
                
                tempFeats = [tempLead(1,tcbeInd) tempLead(1,specInd)];
                
                tempModelData = cell2mat(tempData2) .* (1.01 - cell2mat(tempData1));
                
                modelData{(lead*length(tcbeFeats)*numRandPerms)+perm+(numRandPerms*(tcbe-1)),1} = [tempModelData cell2mat(responseClass(2:end,:)) cell2mat(responseVars(2:end,1))];
                featNames{(lead*length(tcbeFeats)*numRandPerms)+perm+(numRandPerms*(tcbe-1))}(1,:) = tempFeats;
                featNames{(lead*length(tcbeFeats)*numRandPerms)+perm+(numRandPerms*(tcbe-1))}(2,:) = num2cell([tcbeInd specInd]);
            end
        end
        toc
    end
    modelDataOVA{class} = modelData;
    featDataOVA{class} = featNames';
end

rng(0);

close all;
clearvars -except modelDataOVA featDataOVA;
clc;

% save('FeatCombination_ModelData_Extended','modelDataOVA','featDataOVA','-v7.3');

%% KFold Adaption for per Subject

numSubs = 46;

for kfold = 1:numSubs
    
    disp([newline 'Running Models for K-Fold: ' num2str(kfold) newline])
    
    tClass = tic;
    
    for class = 1:5
       
        mdlClassData = modelDataOVA{class};
        
        tic
        
        parfor model = 1:length(mdlClassData)
            
            modelData = mdlClassData{model};

            X_test = modelData(modelData(:,7) == kfold,1:5);
            Y_test = categorical(modelData(modelData(:,7) == kfold,6));
            trueLabelTest = modelData(modelData(:,7) == kfold,end);
            
            tempData = modelData(modelData(:,7) ~= kfold,1:7);
            
            [X,synthClass] = smote(tempData(:,1:5),[],'Class',tempData(:,6));
            
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

save('FeatCombination_Mdl_Outputs_Extended.mat','labelsKFold','scoresKFold','CKFold','resultsKFold','testLabelsKFold','-v7.3');

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
%         indF1 = zeros(5,5);
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
        
        tallyBest = zeros(1,size(resultsKFold{1}{class},2));
        
        for kfold = 1:46
            
            bestMdls = bestMdlsKFold{kfold}{class};
            
            for mdls = 1:length(bestMdls)
                crntMdl = bestMdls(mdls);
                tallyBest(crntMdl) = tallyBest(crntMdl) + 1;
            end
        end
        tallyBestClass(class,:) = tallyBest;
    end
    
    tcbeFeats = 13:21;
    
    for class = 1:5
        
        featsBest = zeros(1,length(tcbeFeats));
        
        for kfold = 1:46
            
            bestMdls = bestMdlsKFold{kfold}{class};
            
            for mdls = 1:length(bestMdls)
                crntMdl = bestMdls(mdls);
                featData = cell2mat(featDataOVA{class}{crntMdl}(2,1));
                
                indice = find(tcbeFeats == featData);
                
                featsBest(indice) = featsBest(indice) + 1;
            end
        end
        featsBestClass(class,:) = featsBest;
        
        clear featsBest
    end
    
    specFeats = 1:12;
    
    for class = 1:5
        
        featsBest = zeros(1,length(specFeats));
        
        for kfold = 1:46
            
            bestMdls = bestMdlsKFold{kfold}{class};
            
            for mdls = 1:length(bestMdls)
                crntMdl = bestMdls(mdls);
                featData = cell2mat(featDataOVA{class}{crntMdl}(2,2:end));
                
%                 indice = find(tcbeFeats == featData);
                
                featsBest(featData) = featsBest(featData) + 1;
            end
        end
        featsBestClassSpec(class,:) = featsBest;
        
        clear featsBest
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
    
    figure;
    cm = confusionchart(finalConfMat,{'TP','TA','TE','TR','OTH'},'Title',{join(["AUC:"; string(round(avgAUC,3)); "ACC:"; string(round(finalResConfMat.Accuracy,3));...
        "F1:"; string(round(finalResConfMat.F1_score,3)); "MCC:"; string(round(finalResConfMat.MatthewsCorrelationCoefficient,3))])});
    sortClasses(cm,{'TP','TA','TE','TR','OTH'})
    
    mdlAUCs(topNum) = avgAUC;
    mdlF1(topNum) = finalResConfMat.F1_score;
    mdlACC(topNum) = finalResConfMat.Accuracy;
    mdlMCC(topNum) = finalResConfMat.MatthewsCorrelationCoefficient;
    
end

% save('FeatCombination_FinalOutputs.mat','scoresCentroidKFold','finalPredictsKFold','finalConfKFold','finalResKFold',...
%     'finalStatsKFold','finalConfMat','finalResConfMat','bestMdlsKFold','bestFeatsKFold','-v7.3');

%%
titles_Plots = {'Task Precursor','Audio Task Interruption','Task Execution','Task Recovery','OTHER'};
% title_Features = {'Spectral Intensity','Temporal Cross Band Entropy','Engagement Index'};

figure;
t = tiledlayout(2,3);
for class = 1:size(tallyBestClass,1)
    
    x = 1:size(resultsKFold{1}{class},2);
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
        plot(x,ones(1,size(resultsKFold{1}{class},2))*top3(i),'-*','LineWidth',2,'Color',colors{i},'MarkerIndices',find(ones(1,size(resultsKFold{1}{class},2))*top3(i) == y))
    end
    
end
linkaxes([ax1,ax2,ax3,ax4,ax5],'xy')

title(t,'Top Models for Each Task State for Single Lead Models')

set(gcf, 'Position', get(0, 'Screensize'));

%%

predictors_names = {'ROC_F', 'ROC_{FD}' 'ROC_{TBA}', 'ROC_{BA}', 'ROC_B' 'ROC_{BE}' ...
    'ROC_{ABT}' 'ROC_{ABTE}' 'ROC_{GBA}'}; 

titles_Plots = {'Task Precursor','Audio Task Interruption','Task Execution','Task Recovery','OTHER'};

figure;
t = tiledlayout(2,3);
for class = 1:size(featsBestClass,1)
    
    x = 1:length(featsBestClass(class,:));
    y = featsBestClass(class,:);
    
    %     ticks = 0:1:max(y);
    
    eval(['ax' num2str(class) ' = nexttile;']);
    bar(x,y)
    title(titles_Plots{class})
    xticklabels(predictors_names)
    xtickangle(45)
    xlabel('Feature')
    ylabel('Number of Appearances in Top 4')
    %     yticks(ticks)
    
    set(gca,'fontweight','bold')
    
    hold on
    
    vals = unique(y);
    top3 = vals(end-2:end);
    
    colors = {[0.4940 0.1840 0.5560],[0.8500 0.3250 0.0980],[0.4660 0.6740 0.1880]};
    
    for i = 1:length(top3)
        plot(x,ones(1,length(featsBestClass(class,:)))*top3(i),'-*','LineWidth',2, ...
            'Color',colors{i},'MarkerIndices',find(ones(1,length(featsBestClass(class,:)))*top3(i) == y))
    end
    
end
linkaxes([ax1,ax2,ax3,ax4,ax5],'xy')

title(t,'Top Features for Each Task State for TCBE Combinations')

set(gcf, 'Position', get(0, 'Screensize'));


%%
predictors_names = {'W_1', 'W_2', 'W_3' 'W_4' 'W_5', 'W_6', 'W_7', 'W_8' 'W_9' 'W_{10}', 'W_{11}' 'W_{12}'};
    
titles_Plots = {'Task Precursor','Audio Task Interruption','Task Execution','Task Recovery','OTHER'};

figure;
t = tiledlayout(2,3);
for class = 1:size(featsBestClassSpec,1)
    
    x = 1:length(featsBestClassSpec(class,:));
    y = featsBestClassSpec(class,:);
    
    %     ticks = 0:1:max(y);
    
    eval(['ax' num2str(class) ' = nexttile;']);
    bar(x,y)
    title(titles_Plots{class})
    xticklabels(predictors_names)
    xtickangle(45)
    xlabel('Feature')
    ylabel('Number of Appearances in Top 4')
    %     yticks(ticks)
    
    set(gca,'fontweight','bold')
    
    hold on
    
    vals = unique(y);
    top3 = vals(end-2:end);
    
    colors = {[0.4940 0.1840 0.5560],[0.8500 0.3250 0.0980],[0.4660 0.6740 0.1880]};
    
    for i = 1:length(top3)
        plot(x,ones(1,length(featsBestClassSpec(class,:)))*top3(i),'-*','LineWidth',2, ...
            'Color',colors{i},'MarkerIndices',find(ones(1,length(featsBestClassSpec(class,:)))*top3(i) == y))
    end
    
end
linkaxes([ax1,ax2,ax3,ax4,ax5],'xy')

title(t,'Top Features for Each Task State for Spectral Intensity Combinations')

set(gcf, 'Position', get(0, 'Screensize'));