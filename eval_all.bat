@echo off

REM ===== 固定使用 py311 环境的 Python =====
set PYTHON=C:\Users\Miranda\.conda\envs\py311\python.exe

echo ==============================
echo Experiment started at %date% %time%
echo Python: %PYTHON%
echo ==============================

echo [01/20] lr   S=10  %date% %time%
"%PYTHON%" -m nsp.scripts.evaluate_model --problem dblrp_10_10 --model lr   --n_scenarios 10 --test_set 0 --n_procs -1

echo [02/20] nn_p S=10  %date% %time%
"%PYTHON%" -m nsp.scripts.evaluate_model --problem dblrp_10_10 --model nn_p --n_scenarios 10 --test_set 0 --n_procs -1

echo [03/20] nn_e S=10  %date% %time%
"%PYTHON%" -m nsp.scripts.evaluate_model --problem dblrp_10_10 --model nn_e --n_scenarios 10 --test_set 0 --n_procs -1

echo [04/20] EF   S=10  %date% %time%
"%PYTHON%" -m nsp.scripts.evaluate_extensive --problem dblrp_10_10 --n_scenarios 10 --test_set 0 --n_procs -1


echo [05/20] lr   S=20  %date% %time%
"%PYTHON%" -m nsp.scripts.evaluate_model --problem dblrp_10_10 --model lr   --n_scenarios 20 --test_set 0 --n_procs -1

echo [06/20] nn_p S=20  %date% %time%
"%PYTHON%" -m nsp.scripts.evaluate_model --problem dblrp_10_10 --model nn_p --n_scenarios 20 --test_set 0 --n_procs -1

echo [07/20] nn_e S=20  %date% %time%
"%PYTHON%" -m nsp.scripts.evaluate_model --problem dblrp_10_10 --model nn_e --n_scenarios 20 --test_set 0 --n_procs -1

echo [08/20] EF   S=20  %date% %time%
"%PYTHON%" -m nsp.scripts.evaluate_extensive --problem dblrp_10_10 --n_scenarios 20 --test_set 0 --n_procs -1


echo [09/20] lr   S=30  %date% %time%
"%PYTHON%" -m nsp.scripts.evaluate_model --problem dblrp_10_10 --model lr   --n_scenarios 30 --test_set 0 --n_procs -1

echo [10/20] nn_p S=30  %date% %time%
"%PYTHON%" -m nsp.scripts.evaluate_model --problem dblrp_10_10 --model nn_p --n_scenarios 30 --test_set 0 --n_procs -1

echo [11/20] nn_e S=30  %date% %time%
"%PYTHON%" -m nsp.scripts.evaluate_model --problem dblrp_10_10 --model nn_e --n_scenarios 30 --test_set 0 --n_procs -1

echo [12/20] EF   S=30  %date% %time%
"%PYTHON%" -m nsp.scripts.evaluate_extensive --problem dblrp_10_10 --n_scenarios 30 --test_set 0 --n_procs -1


echo [13/20] lr   S=40  %date% %time%
"%PYTHON%" -m nsp.scripts.evaluate_model --problem dblrp_10_10 --model lr   --n_scenarios 40 --test_set 0 --n_procs -1

echo [14/20] nn_p S=40  %date% %time%
"%PYTHON%" -m nsp.scripts.evaluate_model --problem dblrp_10_10 --model nn_p --n_scenarios 40 --test_set 0 --n_procs -1

echo [15/20] nn_e S=40  %date% %time%
"%PYTHON%" -m nsp.scripts.evaluate_model --problem dblrp_10_10 --model nn_e --n_scenarios 40 --test_set 0 --n_procs -1

echo [16/20] EF   S=40  %date% %time%
"%PYTHON%" -m nsp.scripts.evaluate_extensive --problem dblrp_10_10 --n_scenarios 40 --test_set 0 --n_procs -1


echo [17/20] lr   S=50  %date% %time%
"%PYTHON%" -m nsp.scripts.evaluate_model --problem dblrp_10_10 --model lr   --n_scenarios 50 --test_set 0 --n_procs -1

echo [18/20] nn_p S=50  %date% %time%
"%PYTHON%" -m nsp.scripts.evaluate_model --problem dblrp_10_10 --model nn_p --n_scenarios 50 --test_set 0 --n_procs -1

echo [19/20] nn_e S=50  %date% %time%
"%PYTHON%" -m nsp.scripts.evaluate_model --problem dblrp_10_10 --model nn_e --n_scenarios 50 --test_set 0 --n_procs -1

echo [20/20] EF   S=50  %date% %time%
"%PYTHON%" -m nsp.scripts.evaluate_extensive --problem dblrp_10_10 --n_scenarios 50 --test_set 0 --n_procs -1


echo ==============================
echo Experiment finished at %date% %time%
echo ==============================
