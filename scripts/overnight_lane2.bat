@echo off
REM Overnight lane 2: thresholds chosen on more validation participants
REM Each run is 3 seeds, compared against the 3-seed reference run.
REM Uses the existing window cache; no database access needed.
REM Uses %%CFG%% from the window; never sets it. Output goes to logs\overnight_lane2.log.
setlocal
if not defined CFG (echo CFG is not set in this window. Set it first, then rerun. & exit /b 1)
if not exist "%CFG%" (echo CFG points to a missing file: %CFG% & exit /b 1)
if not exist logs mkdir logs
set BASE=--config "%CFG%" --threads 8 --monitor auc --learning-rate 0.0003 --repeats 3
set LOG=logs\overnight_lane2.log
echo Started %date% %time% >> %LOG%
echo [%time%] 5 inner participants, 3 seeds
echo ===== 5 inner participants, 3 seeds ===== >> %LOG%
python train_food_intake.py %BASE% --inner-participants 5 >> %LOG% 2>&1
echo [%time%] 8 inner participants, 3 seeds
echo ===== 8 inner participants, 3 seeds ===== >> %LOG%
python train_food_intake.py %BASE% --inner-participants 8 >> %LOG% 2>&1
echo Finished %date% %time% >> %LOG%
echo [%time%] All runs finished. See %LOG%
endlocal
