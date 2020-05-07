@echo off
rem Replace the variables with your github repo url, repo name, test
video name, json named by your UIN
set GIT_REPO_URL=https://github.com/vcskaushik1/Sneeze-Detection.git
set REPO=Sneeze-Detection
set VIDEO=test_video1.mp4
set UINJSON=827004640.json
set UINJPG=827004640.jpg
set JSON=timeLabel.json
set JPG=timeLabel.jpg
git clone %GIT_REPO_URL%
cd %REPO%
echo %VIDEO%
python test.py --video_name %VIDEO%
copy %JPG% %UINJPG%
copy %JSON% %UINJSON%
