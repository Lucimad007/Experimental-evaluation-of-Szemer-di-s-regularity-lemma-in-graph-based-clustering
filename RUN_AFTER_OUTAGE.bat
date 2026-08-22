@echo off
rem ===========================================================================
rem  Resume all small-dataset experiments after a power outage.
rem  SAFE TO DOUBLE-CLICK ANY TIME: every experiment skips work that is
rem  already checkpointed in results\ (per-dataset checkpoints) and simply
rem  continues where it left off.
rem
rem  Scope: the 12 small datasets (n <= 1000). The 5 large datasets
rem  (Segment, Rice, Spambase, Landsat, USPS) need long uninterrupted power;
rem  see the commands at the bottom of this file for when power allows.
rem ===========================================================================
cd /d %~dp0

rem --- guard: never stack duplicate runs on top of live ones ---------------
tasklist /FI "IMAGENAME eq python.exe" 2>NUL | find /I "python.exe" >NUL
if %ERRORLEVEL%==0 (
    echo.
    echo  Experiments are ALREADY RUNNING - not launching duplicates.
    echo  If they are stuck or unwanted, close the minimized windows or run:
    echo      taskkill /F /IM python.exe
    echo  then run this file again.
    echo.
    pause
    exit /b
)

set SMALL=Appendicitis,Wine,Sonar,Seeds,Glass,Thyroid,Spectf,Ecoli,Landmine,Libras,SCC,Raisin

echo [1/5] exp1 full parameter grid - worker 1 (Thyroid, Wine, Glass, Seeds)
start "exp1-w1" /min cmd /c ".venv\Scripts\python.exe -m src.main exp1 --datasets Thyroid,Wine,Glass,Seeds --out results\small_w1"

echo [2/5] exp1 worker 2 (Sonar, Spectf, Landmine, Ecoli)
start "exp1-w2" /min cmd /c ".venv\Scripts\python.exe -m src.main exp1 --datasets Sonar,Spectf,Landmine,Ecoli --out results\small_w2"

echo [3/5] exp1 worker 3 (Appendicitis, Libras)
start "exp1-w3" /min cmd /c ".venv\Scripts\python.exe -m src.main exp1 --datasets Appendicitis,Libras --out results\small_w3"

echo [4/5] exp1 worker 4 (SCC, Raisin)
start "exp1-w4" /min cmd /c ".venv\Scripts\python.exe -m src.main exp1 --datasets SCC,Raisin --out results\small_w4"

echo [5/5] exp2 then exp2b then exp3 on the 12 small datasets
start "exp2-chain" /min cmd /c ".venv\Scripts\python.exe -m src.main exp2 --datasets %SMALL% && .venv\Scripts\python.exe -m src.main exp2b --datasets %SMALL% && .venv\Scripts\python.exe -m src.main exp3 --datasets %SMALL%"

echo.
echo All jobs launched (minimized windows). Progress checkpoints:
echo   results\small_w*\exp1_parameter_influence\raw_runs_partial.csv
echo   results\exp2_enhanced_vs_original\enhanced_vs_original_partial.csv
echo   results\exp2b_regularity_vs_kmeans\regularity_vs_kmeans_partial.csv
echo   results\exp3_vs_recent\table_nmi_partial.csv
echo.
echo Re-run this file after every outage - finished work is skipped.
echo.
rem ---------------------------------------------------------------------------
rem  LARGE DATASETS (run only with stable power; each grid takes hours):
rem  .venv\Scripts\python.exe -m src.main exp1 --datasets Segment --out results\large_w1
rem  .venv\Scripts\python.exe -m src.main exp1 --datasets Rice --out results\large_w2
rem  .venv\Scripts\python.exe -m src.main exp1 --datasets Spambase --out results\large_w3
rem  .venv\Scripts\python.exe -m src.main exp1 --datasets Landsat --out results\large_w4
rem  .venv\Scripts\python.exe -m src.main exp1 --datasets USPS --out results\large_w5
rem  .venv\Scripts\python.exe -m src.main exp2  (all 20 datasets; resumes too)
rem ---------------------------------------------------------------------------
