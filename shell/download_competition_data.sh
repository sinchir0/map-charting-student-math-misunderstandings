#!/bin/bash
kaggle competitions download map-charting-student-math-misunderstandings -p data
sudo apt-get install unzip
unzip data/map-charting-student-math-misunderstandings.zip -d data
