close all;
clear;

PATH_TO_GSW='./gsw_matlab_v3_04';
PATH_TO_AGGR_DATA='../Inputs/';
var2use  = '<PY:VAR2USE>';
folder2use  = '<PY:FOLDER2USE>';
var_tag = '<PY:VAR_TAG>';

addpath(genpath(PATH_TO_GSW));

%% Load data
years = '<PY:YEARS>';
months = '<PY:MONTHS>';
load([PATH_TO_AGGR_DATA, years, months, var_tag '.mat']);

%% Filter out duplicate profiles
switch var2use
    case {'SSS'}
        [C,ia,ic] = unique([profLatAggr,profLongAggr,profJulDayAggr],'rows');
    otherwise
        [C,ia,ic] = unique([profLatAggr',profLongAggr',profJulDayAggr'],'rows');
end

%% Added to normalize the longitude (from 20 to 380)
profLongAggr(profLongAggr<20) = profLongAggr(profLongAggr<20) + 360;

profPresAggrGood = profPresAggr(ia);
switch var2use
    case {'Temperature','Salinity'}
        profTempAggrGood = profTempAggr(ia);
        profPsalAggrGood = profPsalAggr(ia);
    case 'Potential_density'
        profPdensAggrGood = profPdensAggr(ia);
    otherwise
        profDataAggrGood = profDataAggr(ia);
end

profLatAggrGood = profLatAggr(ia);
profLongAggrGood = profLongAggr(ia);
profYearAggrGood = profYearAggr(ia);
profJulDayAggrGood = profJulDayAggr(ia);
profFloatIDAggrGood = profFloatIDAggr(ia);
profCycleNumberAggrGood = profCycleNumberAggr(ia);

%% Starting pressure and end pressure
% Not needed for ML with data at a single level
startPres = cellfun(@min,profPresAggrGood);
endPres = cellfun(@max,profPresAggrGood);

%% Profile selection based on start and end pressure
% Not needed for ML with data at a single level
intStart = <PY:GRID_LOWER>;
intEnd   = <PY:GRID_UPPER>;
selIdx = (startPres >=0 & startPres <= intStart & endPres >= intEnd);

profPresAggrSel = profPresAggrGood(selIdx);
switch var2use
    case {'Temperature','Salinity'}
        profTempAggrSel = profTempAggrGood(selIdx);
        profPsalAggrSel = profPsalAggrGood(selIdx);
    case 'Potential_density'
        profPdensAggrSel = profPdensAggrGood(selIdx);
    otherwise
        profDataAggrSel = profDataAggrGood(selIdx);
end
profLatAggrSel = profLatAggrGood(selIdx);
profLongAggrSel = profLongAggrGood(selIdx);
profYearAggrSel = profYearAggrGood(selIdx);
profJulDayAggrSel = profJulDayAggrGood(selIdx);
profFloatIDAggrSel = profFloatIDAggrGood(selIdx);
profCycleNumberAggrSel = profCycleNumberAggrGood(selIdx);

%% Compute absolute salinity and conservative and potential temperature, you'll need to have the GSW toolbox in Matlab path to run this section, see http://www.teos-10.org/software.htm

% Convert longitude from 20-380 range to 0-360 range
profLongAggrSelTemp = (profLongAggrSel > 360).*(profLongAggrSel - 360) + (profLongAggrSel <= 360).*profLongAggrSel;

switch var2use
    case {'Temperature','Salinity'}
        % Calculate absolute salinity -- needed for ML case
        profAbsSalAggrSel = cellfun(@gsw_SA_from_SP,profPsalAggrSel, profPresAggrSel,...
             num2cell(profLongAggrSelTemp), num2cell(profLatAggrSel), 'UniformOutput', 0);
        % Calculate potential temperature -- needed for ML case
        profPotTempAggrSel = cellfun(@gsw_pt_from_t,profAbsSalAggrSel,profTempAggrSel,...
             profPresAggrSel,'UniformOutput',0);
end

% disp('Uncomment the part to compute pot temp and abs sal and comment the two lines below here')
%profAbsSalAggrSel  = profPsalAggrSel;
%profPotTempAggrSel = profTempAggrSel;

% Prepare variables for vertical interpolation
presGrid = <PY:GRID_LOWER>:<PY:GRID_STRIDE>:<PY:GRID_UPPER>;
nPresGrid = length(presGrid);
nProf = length(profPresAggrSel);
gridVarObsProf = zeros(nProf, nPresGrid);

% parpool(<PY:N_PARPOOL>)
% parfor_progress(nProf);

% Vertical extrapolation on regular grid
for i = 1:nProf
    clear notnan_*
    press=profPresAggrSel{i};
    notnan_press = ~isnan(press);
    switch var2use
        case 'Temperature'
            pottemp=profPotTempAggrSel{i};
            notnan_pottemp = ~isnan(pottemp);
            pottemp = pottemp(notnan_pottemp&notnan_press);
            press = press(notnan_pottemp&notnan_press);
            gridVarObsProf(i, :)=pchip(press, pottemp, presGrid);
        case 'Salinity'
            absS   = profAbsSalAggrSel{i};
            notnan_absS = ~isnan(absS);
            absS = absS(notnan_absS&notnan_press);
            press = press(notnan_absS&notnan_press);
            gridVarObsProf(i, :)=pchip(press, absS, presGrid);
        case 'Potential_density'
            potD   = profPdensAggrSel{i};
            notnan_potD = ~isnan(potD);
            potD = potD(notnan_potD&notnan_press);
            press = press(notnan_potD&notnan_press);
            gridVarObsProf(i, :)=pchip(press, potD, presGrid);
        otherwise
            Data = profDataAggrSel{i};
            notnan_Data = ~isnan(Data);
            Data = Data(notnan_Data&notnan_press);
            press = press(notnan_Data&notnan_press);
            [C,unique_index,ic] = unique([press],'rows');
            Data = Data(unique_index);
            press = press(unique_index);
            if length(press)>3 
                gridVarObsProf(i, :)=pchip(press, Data, presGrid);
            elseif length(press)==1 && length(presGrid)==1 && press==presGrid
                gridVarObsProf(i, :)=Data;
            end
    end
    % parfor_progress;
end
% parfor_progress(0);
%whos profLatAggrSel profLongAggrSel profYearAggrSel profJulDayAggrSel profFloatIDAggrSel ...
%profCycleNumberAggrSel gridVarObsProf intStart intEnd presGrid
clear msk
msk = ~isnan(nansum(gridVarObsProf,2)' + profLatAggrSel + profLongAggrSel + profYearAggrSel + profJulDayAggrSel +...
profFloatIDAggrSel + profCycleNumberAggrSel);
gridVarObsProf = gridVarObsProf(msk,:);
profLatAggrSel = profLatAggrSel(msk);
profLongAggrSel = profLongAggrSel(msk);
profYearAggrSel = profYearAggrSel(msk);
profJulDayAggrSel = profJulDayAggrSel(msk);
profFloatIDAggrSel = profFloatIDAggrSel(msk);
profCycleNumberAggrSel = profCycleNumberAggrSel(msk);
% Save outputs
save([folder2use,'/Outputs/gridArgoProf_',years,months,'_',var2use,'.mat'],'profLatAggrSel','profLongAggrSel',...
            'profYearAggrSel','profJulDayAggrSel','profFloatIDAggrSel',...
            'profCycleNumberAggrSel','gridVarObsProf',...
            'intStart','intEnd','presGrid','-v7.3');

exit;
