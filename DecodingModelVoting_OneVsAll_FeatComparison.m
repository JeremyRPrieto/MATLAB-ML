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
spectralFeats = 1:12;
tcbeFeats = 13:21;
eiFeats = 22:28;
SampEnFeat = 29;

featInds = {spectralFeats;tcbeFeats;eiFeats;SampEnFeat};

rng(seed1);
numRandPerms = 16;
numRandFeats = 5;

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
        
        for feats = 1:3
            
            tempInd = featInds{feats};
            
            for rnd = 1:numRandPerms
                randFeats = randperm(length(tempInd),5);
                
                ind = tempInd(sort(randFeats));
                
                tempData = tempLead(2:end,ind);
                tempFeats = tempLead(1,ind);
                
                modelData{rnd+(lead*numRandPerms),feats} = [tempData responseClass(2:end,:) responseVars(2:end,1)];
                featNames{rnd+(lead*numRandPerms),feats}(1,:) = tempFeats;
                featNames{rnd+(lead*numRandPerms),feats}(2,:) = num2cell(sort(randFeats));
                
            end
%             tempData = tempLead(2:end,tempInd);
%             tempFeats = tempLead(1,tempInd);
%             
%             modelData{lead+1,feats} = [tempData responseClass(2:end,:) responseVars(2:end,1)];
%             featNames{lead+1,feats}(1,:) = tempFeats;
%             featNames{lead+1,feats}(2,:) = num2cell(tempInd);
            
%             modelData{}
%             featNames{} 
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
        featClassData = featDataOVA{class};
        
        for feats = 1:3
            
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

save('FeatComp_Mdl_Outputs.mat','featDataOVA','labelsKFold','scoresKFold','CKFold','resultsKFold','testLabelsKFold','-v7.3');

%% KFolds and Results of Each Feature Set
for topMdl = 3
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
                f1ScoresAll{class}{feats}(:,kfold) = statsMdl(:,2);
                accAll{class}{feats}(:,kfold) = statsMdl(:,1);
            end
            statsClass{class} = statsFeats;
            
        end
        statsKFold{kfold} = statsClass;
    end
    
    for class = 1:5
        for feats = 1:3
            f1ScoreMeans{feats}(:,class) = mean(f1ScoresAll{class}{feats},2,'omitnan');
            accMeans{feats}(:,class) = mean(accAll{class}{feats},2,'omitnan');
        end
    end
    
    for kfold = 1:length(statsKFold)
        resKFold = statsKFold{kfold};
        for class = 1:length(resKFold)
            resClass = resKFold{class};
            for feats = 1:length(resClass)
                resFeats = resClass{feats};
                
                resMdl = resFeats(:,2);
                [tempRank,indRank] = sort(resMdl);
                tempInd = ~isnan(tempRank);
                ranks = indRank(tempInd);
                
                bestMdlsFeats{feats} = ranks(end-topMdl:end);
            end
            bestMdlsClass{class} = bestMdlsFeats;
        end
        bestMdlsKFold{kfold} = bestMdlsClass;
    end
    
    for feats = 1:3
        for class = 1:5
            
            tallyBest = zeros(1,256);
            
            for kfold = 1:46
                
                bestMdls = bestMdlsKFold{kfold}{class}{feats};
                
                for mdls = 1:length(bestMdls)
                    crntMdl = bestMdls(mdls);
                    tallyBest(crntMdl) = tallyBest(crntMdl) + 1;
                end
            end
            tallyBestClass(class,:) = tallyBest;
        end
        tallyBestFeats{feats} = tallyBestClass;
    end
    
    spectralFeats = 1:12;
    tcbeFeats = 13:21;
    eiFeats = 22:28;
    
    featInds = {spectralFeats;tcbeFeats;eiFeats};
    
    for feats = 1:3
        features = featInds{feats};
        for class = 1:5
            
            featsBest = zeros(1,length(features));
            
            for kfold = 1:46
                
                bestMdls = bestMdlsKFold{kfold}{class}{feats};
                
                for mdls = 1:length(bestMdls)
                    crntMdl = bestMdls(mdls);
                    featData = cell2mat(featDataOVA{class}{feats,crntMdl}(2,:));
                    
                    featsBest(featData) = featsBest(featData) + 1;
                end
            end
            featsBestClass(class,:) = featsBest;
            
            clear featsBest
        end
        featsBestFeats{feats} = featsBestClass;
        
        clear featsBestClass
    end
    
    
    for kfold = 1:length(bestMdlsKFold)
        for class = 1:length(bestMdlsKFold{kfold})
            for feats = 1:length(bestMdlsKFold{kfold}{class})
                bestMdls = bestMdlsKFold{kfold}{class}{feats};
                
                for mdls = 1:length(bestMdls)
                    crntMdl = bestMdls(mdls);
                    
                    bestFeats = featDataOVA{class}{feats,crntMdl};
                    labelsBestMdl = labelsKFold{kfold}{class}{feats}{crntMdl};
                    scoresBestMdl = scoresKFold{kfold}{class}{feats}{crntMdl};
                    confMatBestMdl = CKFold{kfold}{class}{feats}{crntMdl};
                    testLabelsBestMdl = testLabelsKFold{kfold}{class}{feats}{crntMdl};
                    statsBestMdl = statsKFold{kfold}{class}{feats}(crntMdl,:);
                    
                    if mdls == 1
                        combScoresBestMdl = scoresBestMdl(:,2);
                        combStatsBestMdl = statsBestMdl;
                    else
                        combScoresBestMdl = [combScoresBestMdl scoresBestMdl(:,2)];
                        combStatsBestMdl = [combStatsBestMdl;statsBestMdl];
                    end
                    
                    tallyBest(crntMdl) = tallyBest(crntMdl) + 1;
                end
                scoresFeats{feats}(:,class) = CentroidMethodGen(combScoresBestMdl);
                statsFeats{feats} = combStatsBestMdl;
            end
            %         scoresClass(:,class) = scoresFeats;
            statsClass{class} = statsFeats;
            
            clear statsFeats
        end
        scoresBestKFold{kfold} = scoresFeats;
        testLabelsBest{kfold} = testLabelsBestMdl;
        statsBestKFold{kfold} = statsClass;
        
        clear scoresFeats statsClass
    end
    
    %
    for class = 1:5
        for feats = 1:3
            mdlFeats = bestMdlsClass{class}{feats};
            for mdls = 1:length(mdlFeats)
                crntMdl = mdlFeats(mdls);
                bestFeatNames{class}{feats}(mdls,:) = featDataOVA{class}{feats,crntMdl}(2,:);
            end
        end
    end
    
    % %% New Organization of Scores
    % for kfold = 1:length(scoresKFold)
    %     for feats = 1:3
    %         for class = 1:5
    %
    %             tempScore = scoresKFold{kfold}{class}{feats};
    %
    %             scoresMat(:,class) = tempScore;
    %
    %         end
    %         scoresFeats{feats} = scoresMat;
    %
    %         scoresKFold_ReOrg{kfold,feats} = scoresMat;
    %         clear scoresMat
    %     end
    % %     scoresKFold_ReOrg{kfold} = scoresFeats;
    % end
    
    for kfold = 1:size(scoresBestKFold,2)
        for feats = 1:size(scoresBestKFold{kfold},2)
            
            [~,finalPredictsKFold{kfold,feats}] = max(scoresBestKFold{kfold}{feats},[],2);
            finalConfKFold{kfold,feats} = confusionmat(testLabelsBest{kfold},finalPredictsKFold{kfold,feats});
            finalResKFold{kfold,feats} = getValues(finalConfKFold{kfold,feats});
            
            finalStatsKFold{kfold,feats}(:,1) = finalResKFold{kfold,feats}.Accuracy;
            finalStatsKFold{kfold,feats}(:,2) = finalResKFold{kfold,feats}.F1_score;
            finalStatsKFold{kfold,feats}(:,3) = finalResKFold{kfold,feats}.MatthewsCorrelationCoefficient;
            
        end
    end
    
    % Final Confusion Matrices per Feature
    finalConfMat_Feats = cell(1,3);
    finalResConfMat_Feats = cell(1,3);
    
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
    
    for res = 1:3
        [aucClass(res,:),finalStatsConfMat{res}(:,4)] = AUC_Calc_MultiClass(finalConfMat_Feats{res},5);
    end
    
    for feats = 1:3
%         figure;
        cm = confusionchart(finalConfMat_Feats{feats},{'TP','TA','TE','TR','OTH'},'Title',{join(["AUC:"; string(round(finalStatsConfMat{feats}(:,4),3)); "ACC:"; string(round(finalStatsConfMat{feats}(:,1),3));...
            "F1:"; string(round(finalStatsConfMat{feats}(:,2),3)); "MCC:"; string(round(finalStatsConfMat{feats}(:,3),3))])});
        sortClasses(cm,{'TP','TA','TE','TR','OTH'})
        
        mdlAUCs(feats,topMdl) = finalStatsConfMat{feats}(:,4);
        mdlF1(feats,topMdl) = finalStatsConfMat{feats}(:,2);
        mdlACC(feats,topMdl) = finalStatsConfMat{feats}(:,1);
        mdlMCC(feats,topMdl) = finalStatsConfMat{feats}(:,3);
        aucClasses{feats}(topMdl,:) = aucClass(feats,:);
        
        %     if feats == 1
        %         saveas(gcf,'Top_15_Models_Spec.svg')
        %         saveas(gcf,'Top_15_Models_Spec.png')
        %     elseif feats == 2
        %         saveas(gcf,'Top_15_Models_TCBE.svg')
        %         saveas(gcf,'Top_15_Models_TCBE.png')
        %     elseif feats == 3
        %         saveas(gcf,'Top_15_Models_EI.svg')
        %         saveas(gcf,'Top_15_Models_EI.png')
        %     end
    end
end

%%
% for i = 1:size(aucClasses,2)
%     
%     temp = aucClasses{i};
%    
%     aucMean(:,i) = mean(temp);
%     aucSE(:,i) = std(temp) / sqrt(length(temp));
%     
% end

% groupedBarsMeanStd(aucMean,aucSE)
bar(aucClass')
ylabel('Mean AUCs','FontWeight','bold')
xlabel('Cognitive Task States','FontWeight','bold')
title('AUC per Class for Top 4 Models')
xticklabels({'TP','TA','TE','TR','OTH'})
legend({'SpecInt','TCBE','EI'},'Location','northwest')

title_Features = {'Spectral Intensity','Temporal Cross Band Entropy','Engagement Index'};

% for feats = 1:3
%     figure;
%     plot(mdlAUCs(feats,:),'-o')
%     hold on
%     plot(mdlF1(feats,:),'-o')
%     plot(mdlACC(feats,:),'-o')
%     plot(mdlMCC(feats,:),'-o')
% %     xline(3,'LineWidth',2)
%     hold off
%     ylim([0 1])
%     xlim([1 topMdl])
%     title(['Calculation of Final Results over Top Models for Feature Set: ' title_Features{feats}])
%     xlabel('Number of Selected Top Models','FontWeight','bold')
% %     ylabel('AUC')
% end



%%
titles_Plots = {'Task Precursor','Audio Task Interruption','Task Execution','Task Recovery','OTHER'};
title_Features = {'Spectral Intensity','Temporal Cross Band Entropy','Engagement Index'};

for feats = 1:3
    figure;
    t = tiledlayout(2,3);
    for class = 1:size(tallyBestFeats{feats},1)
        
        x = 1:256;
        y = tallyBestFeats{feats}(class,:);
        
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
    
    title(t,['Top Models for Each Task State for ' title_Features{feats}])
    
    set(gcf, 'Position', get(0, 'Screensize'));
end

%%
predictors_names = {'W_1', 'W_2', 'W_3' 'W_4' 'W_5', 'W_6', 'W_7', 'W_8' 'W_9' 'W_{10}', 'W_{11}' 'W_{12}' ...
    'ROC_F', 'ROC_{FD}' 'ROC_{TBA}', 'ROC_{BA}', 'ROC_B' 'ROC_{BE}' ...
    'ROC_{ABT}' 'ROC_{ABTE}' 'ROC_{GBA}' 'EI_S' 'EI_M' 'EI_{SD}' 'EIE_S' 'EIE_M' 'EIE_{SD}' 'EI' 'SE' };

spectralFeats = 1:12;
tcbeFeats = 13:21;
eiFeats = 22:28;

featInds = {spectralFeats;tcbeFeats;eiFeats};

titles_Plots = {'Task Precursor','Audio Task Interruption','Task Execution','Task Recovery','OTHER'};
title_Features = {'Spectral Intensity','Temporal Cross Band Entropy','Engagement Index'};

for feats = 1:3
    figure;
    t = tiledlayout(2,3);
    for class = 1:size(featsBestFeats{feats},1)
        
        x = 1:length(featsBestFeats{feats}(class,:));
        y = featsBestFeats{feats}(class,:);
        
        %     ticks = 0:1:max(y);
        
        eval(['ax' num2str(class) ' = nexttile;']);
        bar(x,y)
        title(titles_Plots{class})
        xticklabels(predictors_names(featInds{feats}))
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
            plot(x,ones(1,length(featsBestFeats{feats}(class,:)))*top3(i),'-*','LineWidth',2, ...
                'Color',colors{i},'MarkerIndices',find(ones(1,length(featsBestFeats{feats}(class,:)))*top3(i) == y))
        end

    end
    linkaxes([ax1,ax2,ax3,ax4,ax5],'xy')
    
    title(t,['Top Features for Each Task State for ' title_Features{feats}])
    
    set(gcf, 'Position', get(0, 'Screensize'));
end

% 
% saveas(gcf,'Top_15_Models_LeadCount.svg')
% saveas(gcf,'Top_15_Models_LeadCount.png')


% save('FeatComp_FinalOutputs.mat','finalConfMat_Feats','finalResConfMat_Feats','finalStatsConfMat','scoresKFold_ReOrg','finalPredictsKFold',...
%     'finalConfKFold','finalResKFold','finalStatsKFold','-v7.3')

%% Confusion Matrix Plotting

featNames = {'Spectral Intensity','TCBE','Engagement Index'};

for feat = 1:length(finalConfMat_Feats)
    
    figure;
    
    confMat = finalConfMat_Feats{feat};
    statsMat = finalStatsConfMat{feat};
    
    cm = confusionchart(confMat,{'TP','TA','TE','TR','OTH'},'Title',{join(["AUC:"; string(round(statsMat(4),3)); "ACC:"; string(round(statsMat(1),3)); ...
        "F1:"; string(round(statsMat(2),3)); "MCC:"; string(round(statsMat(3),3))])});
    sortClasses(cm,{'TP','TA','TE','TR','OTH'});
    
    eval(['saveas(gcf,''' featNames{feat} '.svg'')']);
    eval(['saveas(gcf,''' featNames{feat} '.png'')']);
end
    
    