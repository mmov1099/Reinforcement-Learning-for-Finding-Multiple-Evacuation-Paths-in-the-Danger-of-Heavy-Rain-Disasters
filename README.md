# Reinforcement Learning for Finding Multiple Evacuation Paths in the Danger of Heavy Rain Disasters

This repository is my 4th year undergraduate thesis project.  
Reinforcement learning Q-lerning is used to find a route considering the distance from the river, landslide hazard areas, elevation, and the length of the path.  
In this code, a route search is performed for a specific area in Iizuka City, Fukuoka Prefecture.  
The reward r is calculated as a weighted sum of each of them.  
After the routes are calculated, the degree of safety is calculated for each route using an original method and output to a csv file.  
See the PDF file for the detailed calculation method.  

## setup
```bash
brew install gdal
pip install -r requirements.txt
```

## data
Most of the data are from the Geospatial Information Authority of Japan (GSI).  
Please refer to the Appendix for the web pages from which the data were obtained.  
Road information was obtained from NetworkX.  
Only the necessary columns were extracted and compiled into a pickle file.  
The data is not in the repository according to the terms of use of the organization that distributes the data.  

## run
```python
python flood_for_thesis.py
```

## Appedix
[国土数値情報ダウンロードサービス](https://nlftp.mlit.go.jp/ksj/)  
[基盤地図情報ダウンロードサービス](https://fgd.gsi.go.jp/download/mapGis.php?tab=dem)  
[各種資料|基盤地図情報ダウンロードサービス](https://fgd.gsi.go.jp/download/documents.html)  