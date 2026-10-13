@echo off
REM Sequence model: bidirectional GRU over 15 consecutive windows (2 min), 3 seeds,
REM with 5 inner validation participants, so it compares against the 5-inner 3-seed run.
REM Uses %%CFG%% from the window; never sets it. Output goes to logs\sequence_run.log.
setlocal
if not defined CFG (echo CFG is not set in this window. Set it first, then rerun. & exit /b 1)
if not exist "%CFG%" (echo CFG points to a missing file: %CFG% & exit /b 1)
if not exist logs mkdir logs
set BASE=--config "%CFG%" --threads 8 --monitor auc --learning-rate 0.0003 --repeats 3 --inner-participants 5
set LOG=logs\sequence_run.log
echo Started %date% %time% >> %LOG%
echo [%time%] sequence model over 15 windows, 5 inner participants, 3 seeds
echo ===== sequence model over 15 windows, 5 inner, 3 seeds ===== >> %LOG%
python train_food_intake.py %BASE% --sequence 15 >> %LOG% 2>&1
echo Finished %date% %time% >> %LOG%
echo [%time%] All runs finished. See %LOG%
endlocal
