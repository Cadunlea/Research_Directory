@echo off
REM Overnight lane 1: soft labels (train on the share of each window that is eating)
REM Each run is 3 seeds, compared against the 3-seed reference run.
REM The first run rebuilds the windows from the database into ets_cache_soft
REM (Docker must be running) and stops at once if they differ from ets_cache.
REM Uses %%CFG%% from the window; never sets it. Output goes to logs\overnight_lane1.log.
setlocal
if not defined CFG (echo CFG is not set in this window. Set it first, then rerun. & exit /b 1)
if not exist "%CFG%" (echo CFG points to a missing file: %CFG% & exit /b 1)
if not exist logs mkdir logs
set BASE=--config "%CFG%" --threads 8 --monitor auc --learning-rate 0.0003 --repeats 3
set LOG=logs\overnight_lane1.log
echo Started %date% %time% >> %LOG%
echo [%time%] soft labels, 3 seeds (rebuilds the window cache first)
echo ===== soft labels, 3 seeds ===== >> %LOG%
python train_food_intake.py %BASE% --soft-labels --rebuild-cache --cache-dir ets_cache_soft >> %LOG% 2>&1
echo [%time%] soft labels + 5 inner participants, 3 seeds
echo ===== soft labels + 5 inner participants, 3 seeds ===== >> %LOG%
python train_food_intake.py %BASE% --soft-labels --cache-dir ets_cache_soft --inner-participants 5 >> %LOG% 2>&1
echo Finished %date% %time% >> %LOG%
echo [%time%] All runs finished. See %LOG%
endlocal
