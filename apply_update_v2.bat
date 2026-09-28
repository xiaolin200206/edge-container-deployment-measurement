@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

rem ============================================================
rem  Applies the v2.0-tsusc update to your local repository,
rem  commits, pushes, tags, and checks the result.
rem  Double-click this file inside the unzipped repo-update-v2 folder.
rem ============================================================

set "DST=C:\Users\Lim Ding Shan\Desktop\Durian project and paper\Third paper\github"
set "SRC=%~dp0"

if not exist "%DST%\.git" (
  echo [STOP] No git repository at:
  echo        %DST%
  echo        Open this .bat in Notepad and correct the DST line.
  pause & exit /b 1
)
if not exist "%SRC%three_condition\analysis\analyse_v2.py" (
  echo [STOP] This .bat must stay inside the unzipped repo-update-v2 folder.
  pause & exit /b 1
)

cd /d "%DST%"
echo.
echo   update from : %SRC%
echo   repository  : %DST%
echo.

echo == Syncing with GitHub first ==
git pull --ff-only || (echo [STOP] git pull failed - send this window to Claude & pause & exit /b 1)

echo.
echo == Copying files ==
if exist "%DST%\three_condition" rmdir /S /Q "%DST%\three_condition"
robocopy "%SRC%three_condition" "%DST%\three_condition" /E /NFL /NDL /NJH /NJS /NP >nul
if errorlevel 8 (echo [STOP] copy failed & pause & exit /b 1)
copy /Y "%SRC%README.md" "%DST%\README.md" >nul
copy /Y "%SRC%CHANGELOG.md" "%DST%\CHANGELOG.md" >nul
echo    three_condition\, README.md, CHANGELOG.md

echo.
echo == Removing superseded root Dockerfiles ==
git rm --quiet --ignore-unmatch Dockerfile.matched Dockerfile.legacy
echo    Dockerfile.matched, Dockerfile.legacy removed (now in three_condition\runner)

echo.
echo == Checking the copy ==
set "FAIL="
if not exist "%DST%\three_condition\runs\native_3\basil_data.csv.gz" (echo    [MISSING] run data & set FAIL=1)
if not exist "%DST%\three_condition\manuscript\main.pdf" (echo    [MISSING] manuscript & set FAIL=1)
findstr /C:"22857310" "%DST%\README.md" >nul || (echo    [MISSING] new DOI in README & set FAIL=1)
if defined FAIL (pause & exit /b 1)
echo    all checks passed

echo.
echo == Committing ==
git add -A
git commit -F "%SRC%COMMIT_MSG.txt" || (echo [STOP] commit failed & pause & exit /b 1)

echo.
echo == Pushing ==
git push || (echo [STOP] push failed & pause & exit /b 1)

echo.
echo == Tagging v2.0-tsusc ==
git tag -a v2.0-tsusc -m "Version cited in the TSUSC submission"
git push origin v2.0-tsusc

echo.
echo == Result ==
git log --oneline -1
git ls-remote --tags origin v2.0-tsusc
echo.
echo If the last line shows refs/tags/v2.0-tsusc, the push worked.
pause
