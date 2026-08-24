@echo off
rem ===========================================================================
rem  Resume the remaining small-dataset work. Double-click after any outage.
rem  Nothing is recomputed: each experiment skips datasets already present in
rem  its checkpoint file.
rem
rem  These windows are DETACHED from any editor/agent session, so they survive
rem  even if the editor closes. Close the windows (or run taskkill /F /IM
rem  python.exe) to stop them.
rem ===========================================================================
cd /d %~dp0

tasklist /FI "IMAGENAME eq python.exe" 2>NUL | find /I "python.exe" >NUL
if %ERRORLEVEL%==0 (
    echo.
    echo  Experiments are ALREADY RUNNING - not launching duplicates.
    echo  To stop them:  taskkill /F /IM python.exe
    echo.
    pause
    exit /b
)

set SMALL=Appendicitis,Wine,Sonar,Seeds,Glass,Thyroid,Spectf,Ecoli,Landmine,Libras,SCC,Raisin

echo Launching 5 detached jobs (4 exp1 grid workers + exp3)...

start "exp1-w1 Seeds"          /min cmd /c ".venv\Scripts\python.exe -u -m src.main exp1 --datasets Thyroid,Wine,Glass,Seeds --out results\small_w1 > results\log_w1.txt 2>&1"
start "exp1-w2 Spectf,Ecoli"   /min cmd /c ".venv\Scripts\python.exe -u -m src.main exp1 --datasets Sonar,Spectf,Landmine,Ecoli --out results\small_w2 > results\log_w2.txt 2>&1"
start "exp1-w3 Libras"         /min cmd /c ".venv\Scripts\python.exe -u -m src.main exp1 --datasets Appendicitis,Libras --out results\small_w3 > results\log_w3.txt 2>&1"
start "exp1-w4 SCC,Raisin"     /min cmd /c ".venv\Scripts\python.exe -u -m src.main exp1 --datasets SCC,Raisin --out results\small_w4 > results\log_w4.txt 2>&1"
start "exp3"                   /min cmd /c ".venv\Scripts\python.exe -u -m src.main exp3 --datasets %SMALL% > results\log_exp3.txt 2>&1"

echo.
echo Done. Watch progress with:
echo     type results\log_w1.txt
echo     type results\log_exp3.txt
echo.
echo When everything finishes, verify the paper's claims with:
echo     .venv\Scripts\python.exe -m proof.verify_claims
echo     .venv\Scripts\python.exe -m src.make_results
echo.
rem ---------------------------------------------------------------------------
rem  exp2 / exp2b on the 12 small datasets are already COMPLETE
rem  (results\exp2_enhanced_vs_original\, results\exp2b_regularity_vs_kmeans\).
rem
rem  LARGE DATASETS (only with stable power; hours each):
rem  .venv\Scripts\python.exe -m src.main exp1 --datasets Segment  --out results\large_w1
rem  .venv\Scripts\python.exe -m src.main exp1 --datasets Rice     --out results\large_w2
rem  .venv\Scripts\python.exe -m src.main exp1 --datasets Spambase --out results\large_w3
rem  .venv\Scripts\python.exe -m src.main exp1 --datasets Landsat  --out results\large_w4
rem  .venv\Scripts\python.exe -m src.main exp1 --datasets USPS     --out results\large_w5
rem ---------------------------------------------------------------------------
