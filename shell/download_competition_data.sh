#!/bin/bash
kaggle competitions download map-probing-questiontext-unique -p data
sudo apt-get install unzip
unzip data/map-probing-questiontext-unique.zip -d data
