@echo off
REM Lane 2: seed repeats, false positives, Transformer, ensemble, ResNet
REM Leave-one-subject-out versions of every comparison, for the paper.
REM Uses %%CFG%% from the window; never sets it. Output goes to logs\weekend_lane2.log.
setlocal
if not defined CFG (echo CFG is not set in this window. Set it first, then rerun. & exit /b 1)
if not exist "%CFG%" (echo CFG points to a missing file: %CFG% & exit /b 1)
if not exist logs mkdir logs
set BASE=--config "%CFG%" --threads 8 --monitor auc --learning-rate 0.0003
set LOG=logs\weekend_lane2.log
echo Started %date% %time% >> %LOG%
echo [%time%] current model, 3 seeds
echo ===== current model, 3 seeds ===== >> %LOG%
python train_food_intake.py %BASE% --repeats 3 --tag seeds >> %LOG% 2>&1
echo [%time%] false-alarm budget 15/h
echo ===== false-alarm budget 15/h ===== >> %LOG%
python train_food_intake.py %BASE% --threshold-objective false-alarms --max-false-alarms 15 >> %LOG% 2>&1
echo [%time%] F0.5 threshold
echo ===== F0.5 threshold ===== >> %LOG%
python train_food_intake.py %BASE% --threshold-objective f0.5 >> %LOG% 2>&1
echo [%time%] 5 inner participants
echo ===== 5 inner participants ===== >> %LOG%
python train_food_intake.py %BASE% --inner-participants 5 >> %LOG% 2>&1
echo [%time%] 1 Transformer layer
echo ===== 1 Transformer layer ===== >> %LOG%
python train_food_intake.py %BASE% --transformer-layers 1 >> %LOG% 2>&1
echo [%time%] augmentation
echo ===== augmentation ===== >> %LOG%
python train_food_intake.py %BASE% --augment >> %LOG% 2>&1
echo [%time%] 3-model ensemble
echo ===== 3-model ensemble ===== >> %LOG%
python train_food_intake.py %BASE% --ensemble 3 >> %LOG% 2>&1
echo [%time%] ResNet baseline
echo ===== ResNet baseline ===== >> %LOG%
python train_food_intake.py %BASE% --architecture resnet >> %LOG% 2>&1
echo Finished %date% %time% >> %LOG%
echo All runs finished. See %LOG%
endlocal
