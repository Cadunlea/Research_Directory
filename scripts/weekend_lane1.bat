@echo off
REM Lane 1: input representation, sampling rate, window, ablations, FCN
REM Leave-one-subject-out versions of every comparison, for the paper.
REM Uses %%CFG%% from the window; never sets it. Output goes to logs\weekend_lane1.log.
setlocal
if not defined CFG (echo CFG is not set in this window. Set it first, then rerun. & exit /b 1)
if not exist "%CFG%" (echo CFG points to a missing file: %CFG% & exit /b 1)
if not exist logs mkdir logs
set BASE=--config "%CFG%" --threads 8 --monitor auc --learning-rate 0.0003
set LOG=logs\weekend_lane1.log
echo Started %date% %time% >> %LOG%
echo [%time%] FFT input
echo ===== FFT input ===== >> %LOG%
python train_food_intake.py %BASE% --input fft >> %LOG% 2>&1
echo [%time%] 32 Hz
echo ===== 32 Hz ===== >> %LOG%
python train_food_intake.py %BASE% --sample-rate 32 >> %LOG% 2>&1
echo [%time%] 64 Hz
echo ===== 64 Hz ===== >> %LOG%
python train_food_intake.py %BASE% --sample-rate 64 >> %LOG% 2>&1
echo [%time%] 16 Hz
echo ===== 16 Hz ===== >> %LOG%
python train_food_intake.py %BASE% --sample-rate 16 >> %LOG% 2>&1
echo [%time%] 16 s windows
echo ===== 16 s windows ===== >> %LOG%
python train_food_intake.py %BASE% --window-seconds 16 >> %LOG% 2>&1
echo [%time%] 4 s windows
echo ===== 4 s windows ===== >> %LOG%
python train_food_intake.py %BASE% --window-seconds 4 >> %LOG% 2>&1
echo [%time%] no channel attention
echo ===== no channel attention ===== >> %LOG%
python train_food_intake.py %BASE% --no-channel-attention >> %LOG% 2>&1
echo [%time%] average pooling
echo ===== average pooling ===== >> %LOG%
python train_food_intake.py %BASE% --pooling average >> %LOG% 2>&1
echo [%time%] no chew head
echo ===== no chew head ===== >> %LOG%
python train_food_intake.py %BASE% --chew-weight 0 >> %LOG% 2>&1
echo [%time%] chew weight 1.0
echo ===== chew weight 1.0 ===== >> %LOG%
python train_food_intake.py %BASE% --chew-weight 1.0 >> %LOG% 2>&1
echo [%time%] FCN baseline
echo ===== FCN baseline ===== >> %LOG%
python train_food_intake.py %BASE% --architecture fcn >> %LOG% 2>&1
echo Finished %date% %time% >> %LOG%
echo All runs finished. See %LOG%
endlocal
