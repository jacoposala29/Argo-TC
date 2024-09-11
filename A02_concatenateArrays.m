% Import variables
years = '<PY:YEARS>';
var2use  = '<PY:VAR2USE>';
folder2use  = '<PY:FOLDER2USE>';
months = '(01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12)';

prefix = strcat(folder2use,'/Outputs/gridArgoProf');

% Work on years to make the loop work
years_vec = strrep(years,'(','');
years_vec = strrep(years_vec,')','');
years_vec = strrep(years_vec,' ','');

years_vec = strsplit(years_vec,',');
if isempty(years_vec{end})
   years_vec = years_vec(1:end-1);
end

% Work on months to make the loop work
months_vec = strrep(months,'(','');
months_vec = strrep(months_vec,')','');
months_vec = strrep(months_vec,' ','');

months_vec = strsplit(months_vec,',');
if isempty(months_vec{end})
   months_vec = months_vec(1:end-1);
end

% Open first file
bfr = load([prefix, '_', years_vec{1}, months_vec{1}, '_', var2use, '.mat']);
% Loop to open all other files
for i=1:length(years_vec)
    for j=1:length(months_vec)
        % Skip first case as it's already loaded before the external loop
        if i == 1 && j == 1
            continue
        end       
        clear B
        % Load another file
        B = load([prefix, '_', years_vec{i}, months_vec{j}, '_', var2use, '.mat']);

        % Concatenate with what we had so far
        gridVarObsProf = cat(1, bfr.gridVarObsProf, B.gridVarObsProf);
        profLatAggrSel = cat(2, bfr.profLatAggrSel, B.profLatAggrSel);
        profLongAggrSel = cat(2, bfr.profLongAggrSel, B.profLongAggrSel);
        profYearAggrSel = cat(2, bfr.profYearAggrSel, B.profYearAggrSel);
        profJulDayAggrSel = cat(2, bfr.profJulDayAggrSel, B.profJulDayAggrSel);
        profFloatIDAggrSel = cat(2, bfr.profFloatIDAggrSel, B.profFloatIDAggrSel);
        profCycleNumberAggrSel = cat(2, bfr.profCycleNumberAggrSel, B.profCycleNumberAggrSel);
        presGrid=bfr.presGrid;
        intStart=bfr.intStart;
        intEnd  =bfr.intEnd;

        % Save what has been concatenated so far...
        save([folder2use '/Outputs/bfr' var2use '.mat'], 'profLatAggrSel','profLongAggrSel','profYearAggrSel',...
                    'profJulDayAggrSel','profFloatIDAggrSel','profCycleNumberAggrSel',...
                    'gridVarObsProf','intStart','intEnd','presGrid', '-v7.3');
        % ... and use it as starting point for the next iteration
        clear bfr
        bfr = load([folder2use '/Outputs/bfr' var2use '.mat']);
    end
end

% At the very end, save file with all concatenated data
save([folder2use '/Outputs/gridArgoProf' var2use '.mat'], 'profLatAggrSel','profLongAggrSel','profYearAggrSel',...
            'profJulDayAggrSel','profFloatIDAggrSel','profCycleNumberAggrSel',...
            'gridVarObsProf','intStart','intEnd','presGrid', '-v7.3');

exit;
