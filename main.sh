#! /bin/bash
unzip -d ./data/Weibo ./data/Weibo/weibotree.txt.zip
pip install -r requirements.txt
# Generate graph data and store in /data/Weibograph
python ./Process/getWeibograph.py
#Generate graph data and store in /data/Twitter15graph
python ./Process/getTwittergraph.py Twitter15
#Generate graph data and store in /data/Twitter16graph
python ./Process/getTwittergraph.py Twitter16
python ./model/Weibo/BiGCN_Weibo.py 1
python ./model/Twitter/BiGCN_Twitter.py Twitter15 1
python ./model/Twitter/BiGCN_Twitter.py Twitter16 1
