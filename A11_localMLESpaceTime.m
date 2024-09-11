close all;
clear;

disp('A11 started')

windowSize = <PY:WINDOW_SIZE>;%8
minNumberOfObs = <PY:MIN_NUM_OBS>;%20;
% region = '_WesternPacific';
region = '<PY:OCEAN_BASIN>';
folder2use  = '<PY:FOLDER2USE>';

month = <PY:CENTER_MONTH>;%9

current_layer = <PY:CURRENT_LAYER>;
startYear = <PY:START_YEAR>;%2007
endYear = <PY:END_YEAR>;%2018
depthIdx = <PY:DEPTH_INDEX>;

[latGrid,longGrid] = <PY:OB_MESHGRID>;

nGrid = numel(latGrid);
nYear = endYear - startYear + 1;

thetasOpt = zeros(size(latGrid));
thetaLatOpt = zeros(size(latGrid));
thetaLongOpt = zeros(size(latGrid));
thetatOpt = zeros(size(latGrid));
sigmaOpt = zeros(size(latGrid));
nll = zeros(size(latGrid));
nResGrid = zeros(size(latGrid));

% Amended to smaller window size
windowSizeGP = <PY:WINDOW_SIZE_GP>;

% Discard previous iterIdx, if it exists
fileName = ['iterIdxLocalMLESpaceTime'...
    '_',num2str(windowSize),'_',num2str(minNumberOfObs),'_',...
    num2str(windowSizeGP),'_',num2str(month,'%02d'),'_',...
    num2str(startYear),'_',num2str(endYear),region,'.txt'];
fileID = fopen(fileName,'w');
fclose(fileID);

parpool(<PY:N_PARPOOL>);

parfor iGrid = 1:nGrid
    
    fileID = fopen(fileName,'a');
    %fprintf(fileID,'%d \n',iGrid);
    fclose(fileID);
    
    predLat = latGrid(iGrid);
    predLong = longGrid(iGrid);

    latMin = predLat - windowSizeGP;
    latMax = predLat + windowSizeGP;
    longMin = predLong - windowSizeGP;
    longMax = predLong + windowSizeGP;

    profLatAggr = cell(1,nYear);
    profLongAggr = cell(1,nYear);
    profJulDayAggr = cell(1,nYear);
    DataResAggr = cell(1,nYear);

    for iYear = startYear:endYear

        if strcmp(region, '_AllBasins')
            S = load([folder2use '/Outputs/Extended/gridArgoRes_',num2str(windowSize),...
                '_',num2str(minNumberOfObs),'_Month_',num2str(month,'%02d'),...
                '_',num2str(iYear), '_extended.mat']);
        else 
            S = load([folder2use '/Outputs/Extended/gridArgoRes_',num2str(windowSize),...
                '_',num2str(minNumberOfObs),'_Month_',num2str(month,'%02d'),...
                '_',num2str(iYear),'_extended_filtered',region,'.mat']);
        end
   
        profLat3Months = S.profLatAggr3Months;
        profLong3Months = S.profLongAggr3Months;
        profJulDay3Months = S.profJulDayAggr3Months;
        DataRes3Months = S.gridDataRes3Months(:, depthIdx)';
        
        % Enable wrap around by duplicating boundary data
        leftBoundaryIdx = find(profLong3Months <= 20 + windowSizeGP);
        rightBoundaryIdx = find(profLong3Months >= 380 - windowSizeGP);
        profLong3Months = [profLong3Months ...
            profLong3Months(leftBoundaryIdx) + 360 ...
            profLong3Months(rightBoundaryIdx) - 360];
        profLat3Months = [profLat3Months ...
            profLat3Months(leftBoundaryIdx) ...
            profLat3Months(rightBoundaryIdx)];
        profJulDay3Months = [profJulDay3Months ...
            profJulDay3Months(leftBoundaryIdx) ...
            profJulDay3Months(rightBoundaryIdx)];
        DataRes3Months = [DataRes3Months ...
            DataRes3Months(leftBoundaryIdx) ...
            DataRes3Months(rightBoundaryIdx)];

        idx = find(profLat3Months > latMin ...
            & profLat3Months < latMax ...
            & profLong3Months > longMin ...
            & profLong3Months < longMax);

        profLatAggr{iYear-startYear+1} = profLat3Months(idx)';
        profLongAggr{iYear-startYear+1} = profLong3Months(idx)';
        profJulDayAggr{iYear-startYear+1} = profJulDay3Months(idx)';
        DataResAggr{iYear-startYear+1} = DataRes3Months(idx)';

    end
    
    nResGrid(iGrid) = sum(cellfun(@length,DataResAggr));
    
    if nResGrid(iGrid) == 0 % No observations in the window
        
        thetasOpt(iGrid) = NaN;
        thetaLatOpt(iGrid) = NaN;
        thetaLongOpt(iGrid) = NaN;
        thetatOpt(iGrid) = NaN;
        sigmaOpt(iGrid) = NaN;
        nll(iGrid) = NaN;
        
        continue;
    end

    try
        fun = @(params) negLogLikSpaceTimeExpGeom_vec(params,...
            profLatAggr,profLongAggr,profJulDayAggr,DataResAggr);
        
        logThetasInit = log(400^2);
        logThetaLatInit = log(5);
        logThetaLongInit = log(5);
        logThetatInit = log(5);
        logSigmaInit = log(100);
        
        opts = optimoptions(@fminunc,'Algorithm','quasi-newton',...
            'MaxFunctionEvaluations',1000,'Display','off');

        [paramOpt,nll(iGrid)] = fminunc(fun,[logThetasInit, logThetaLatInit,...
            logThetaLongInit, logThetatInit, logSigmaInit],opts);
        
        thetasOpt(iGrid) = exp(paramOpt(1));
        thetaLatOpt(iGrid) = exp(paramOpt(2));
        thetaLongOpt(iGrid) = exp(paramOpt(3));
        thetatOpt(iGrid) = exp(paramOpt(4));
        sigmaOpt(iGrid) = exp(paramOpt(5));

    catch
        warning('Optimization failed!');

        thetasOpt(iGrid) = NaN;
        thetaLatOpt(iGrid) = NaN;
        thetaLongOpt(iGrid) = NaN;
        thetatOpt(iGrid) = NaN;
        sigmaOpt(iGrid) = NaN;
        nll(iGrid) = NaN;
    end
    
end

save([folder2use '/Outputs/localMLESpaceTime_Depth_',...
        num2str(current_layer,'%03d'),'_',...
        num2str(windowSize),'_',num2str(minNumberOfObs),'_',...
        num2str(windowSizeGP),'_',num2str(month,'%02d'),'_',...
        num2str(startYear),'_',num2str(endYear),region,'.mat'],...
    'latGrid','longGrid','thetasOpt','thetaLatOpt','thetaLongOpt',...
    'thetatOpt','sigmaOpt','nll','nResGrid','-v7.3');
exit;
