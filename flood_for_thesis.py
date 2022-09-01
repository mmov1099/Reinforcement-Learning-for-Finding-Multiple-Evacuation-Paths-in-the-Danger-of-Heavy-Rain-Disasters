import geopandas as gpd
import pandas as pd
import networkx as nx
import numpy as np
from shapely.geometry import Point
import faiss
import time
from geopy.distance import geodesic
import pickle
import random
#緯度，経度を受け取ってnodeリストの中の最近傍点のインデックスを返す．
def find_nearest_node_index(lat, long, node_lat, node_long):
  xb = list(zip(node_lat, node_long))
  xb = np.array(xb)
  xq = np.array([[lat, long]])
  xb = xb.astype('float32')
  xq = xq.astype('float32')
  index = faiss.IndexFlatL2(xb.shape[1])   # build the index
  index.add(xb)                  # add vectors to the index
  _, result = index.search(xq, k=1)     # actual search
  return result[0][0]
#緯度，経度を受け取ってgdfの中の最近傍点から一番近いnodeリストの中の最近傍点のosmidと避難所のindexを返す．
def find_nearest_shelter(lat, long, gdf, num, node_lat, node_long, osmid):
  xb = np.empty((0,2), float)
  l_y = gdf['geometry'].y.values.tolist()
  l_x = gdf['geometry'].x.values.tolist()
  xb = list(zip(l_y, l_x))
  xb = np.array(xb)
  xq = np.array([[lat, long]])
  xb = xb.astype('float32')
  xq = xq.astype('float32')
  index = faiss.IndexFlatL2(xb.shape[1])   # build the index
  index.add(xb)                  # add vectors to the index
  _, result = index.search(xq, k = num)     # actual search
  shelter = np.empty((0, num), int)
  for x in range(num):
    shelter = np.append(shelter, osmid[find_nearest_node_index(xb[result[0][x]][0], xb[result[0][x]][1], node_lat, node_long)])
  return shelter, result[0]
#道路ネットワーク，開始osmid，終了osmid，全体のedge，nodeを受け取って最短経路のedgeとnodeと経路の長さを返す
def astar(G, start, end, gdf_edge, gdf_node):
  path = nx.astar_path(G, start, end) #最短経路探索
  weight = nx.astar_path_length(G, start, end) #経路の長さを計算
  gdf_path_edge, gdf_path_node = l_to_gdf(path, gdf_edge, gdf_node) #listからgdfに変換
  return gdf_path_edge, gdf_path_node, weight
#osmidのlistをedgeとnodeのgdfに変換する
def l_to_gdf(path, gdf_edge, gdf_node):
  gdf_path_node = gpd.GeoDataFrame()
  for x in range (len(path)):
      gdf_path_node = gdf_path_node.append(gdf_node[gdf_node['osmid'] == path[x]])
  gdf_path_edge = gpd.GeoDataFrame()
  for x in range (len(path) - 1):
      gdf_path_edge = gdf_path_edge.append(gdf_edge[(gdf_edge['u'] == path[x]) & (gdf_edge['v'] == path[x + 1])])
  return gdf_path_edge.reset_index(drop = True), gdf_path_node.reset_index(drop = True)
#gdfからNetworkXの有向グラフに変換
def gdf_to_network(gdf_edge, gdf_node):
  node = gdf_node['osmid']
  edge = gdf_edge['u']
  edge = pd.concat([edge,gdf_edge['v']], axis = 1)
  edge = pd.concat([edge,gdf_edge['length']], axis = 1)
  l_node = node.values.tolist()
  l_edge = edge.values.tolist()
  # グラフの定義
  G = nx.DiGraph()
  G.add_nodes_from(l_node)
  G.add_weighted_edges_from(l_edge)
  return G
#listに含まれる要素xのインデックスを複数返す
def my_index_multi(l, x):
    return [i for i, _x in enumerate(l) if _x == x]
#v1とv2のcos類似度を計算
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
#LineStringの短い字直線の両側にポイントを置く
def line_to_points(line):
  myid_list = line.index.to_list()
  repeat_list = [len(line.coords) for line in line['geometry'].unary_union] #how many points in each Linestring
  coords_list = [line.coords for line in line['geometry'].unary_union]
  points = gpd.GeoDataFrame(columns=['myid', 'order', 'geometry'])
  for myid, repeat, coords in zip(myid_list, repeat_list, coords_list):
      index_num = points.shape[0]
      for i in range(repeat):
          points.loc[index_num+i, 'geometry'] = Point(coords[i])
          points.loc[index_num+i, 'myid'] = myid
  points['order'] = range(1, 1+len(points))
  #you can use groupby method
  points.groupby('myid')['geometry'].apply(list)
  return points
#LineStringをinterval間隔に区切る
def edge_to_points(edge):
  points = line_to_points(edge)
  points = points.drop(['myid', 'order'], axis = 1)
  #重複した要素を消す
  for x in range(len(points)-1):
    if (points.at[x,'geometry'].y == points.at[x+1,'geometry'].y) & (points.at[x,'geometry'].x == points.at[x+1,'geometry'].x):
      points = points.drop(x+1).reset_index(drop = True)
    if len(points)-2 == x:
      break
  lat = points['geometry'].y
  long = points['geometry'].x
  dist = []
  interval = 5 #評価点を置く間隔[m]
  #linestringをハラバラにした時のpointの間隔[m]を算出
  for x in range(len(lat)-1):
    dist.append(geodesic((lat[x], long[x]),(lat[x+1], long[x+1])).m)
  rt_points = gpd.GeoDataFrame(columns = ['geometry']) #returnするポイント
  #intervalの間隔で評価点を置く
  for x in range(len(points) - 1):
    diff_y = (points.at[x + 1, 'geometry'].y - points.at[x, 'geometry'].y) / dist[x] * interval
    diff_x = (points.at[x + 1, 'geometry'].x - points.at[x, 'geometry'].x) / dist[x] * interval
    rt_points = rt_points.append(points[x:x+1])
    df = pd.DataFrame({'Latitude': [points.at[x, 'geometry'].y + diff_y * (1 + z) for z in range(int(dist[x] / interval) - 1)],'Longitude': [points.at[x, 'geometry'].x + diff_x * (1 + z) for z in range(int(dist[x] / interval) - 1)]})
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(df.Longitude, df.Latitude))
    rt_points = rt_points.append(gdf)
  rt_points = rt_points.append(points[x+1:x+2])
  rt_points = rt_points.reset_index(drop = True)
  return rt_points
#pointsについて評価値を計算してリターンする
def eval(points, river_lat, river_long, landslide_lat, landslide_long, height):
  for x in range(len(points)):
    lat = points.at[x, 'geometry'].y
    long = points.at[x, 'geometry'].x
    nearest_node_index = find_nearest_node_index(lat, long, river_lat, river_long)
    points.at[x, 'dist_river'] = geodesic((river_lat[nearest_node_index], river_long[nearest_node_index]), (lat, long)).m
    nearest_node_index = find_nearest_node_index(lat, long, landslide_lat, landslide_long)
    points.at[x, 'dist_landslide'] = geodesic((landslide_lat[nearest_node_index], landslide_long[nearest_node_index]), (lat, long)).m
    focused_height = height[(height['lat'] < lat+0.0001) & (height['lat'] > lat-0.0001) & (height['long'] < long+0.0001) & (height['long'] > long-0.0001)].reset_index(drop = True)
    nearest_node_index = find_nearest_node_index(lat, long, focused_height['lat'].tolist(), focused_height['long'].tolist())
    points.at[x, 'height'] = focused_height.at[nearest_node_index, 'height']
  return points

#強化学習
#シミュレータクラスの設定
class MyEnvironmentSimulator():
  #環境開始時の初期化
  def __init__(self, u, v, init_pos, node_lat, node_long, osmid, dist_end, dist_river, dist_landslide, num, dist, river_mean, landslide_mean, height_mean):
    self._u = u
    self._v = v
    self._lat = node_lat
    self._long = node_long
    self._osmid = osmid
    #height，dist_river，dist_endはosmidと対応づけて格納する
    self._end = dist_end
    self._river = dist_river
    self._landslide = dist_landslide
    #避難開始地点の避難先からの道のり
    self._a = dist
    self._river_mean = river_mean
    self._landslide_mean = landslide_mean
    self._height_mean = height_mean
    #一番最初の強化学習の時だけ初期化
    if num == 0:
      self._end = [0] * len(self._end)
      self._river = [0] * len(self._river)
      self._landslide = [0] * len(self._landslide) 
      self._end[osmid.index(init_pos)] = dist
    self.reset(init_pos)
  #エピソード終了時の初期化
  def reset(self, init_pos):
    self._state = init_pos
    self._dist_end = float('inf')
    self._dist_river = 0
    self._height = 0
    self._dist_landslide = float('inf')
    return self._state
  #行動による状態変化
  def step(self, action_index, dest, G, river_lat, river_long, l_height, landslide_lat, landslide_long):
    #self._stateに行動後の状態を代入
    self._state = self._v[action_index]
    index = self._osmid.index(self._state)
    #self._stateのself._osmid上のインデックスからheight，dist_river，dist_endを代入
    height = l_height[index]
    dist_river = self._river[index]
    dist_end = self._end[index]
    dist_landslide = self._landslide[index]
    y = self._lat[index]
    x = self._long[index]
    #dist_river，dist_landslide，dist_endが0なら計算してlistに格納
    if dist_landslide == 0:
      nearest_node_index = find_nearest_node_index(y, x, landslide_lat, landslide_long)
      self._landslide[index] = geodesic((landslide_lat[nearest_node_index], landslide_long[nearest_node_index]), (y, x)).m
      dist_landslide = self._landslide[index]
    if dist_river == 0:
      nearest_node_index = find_nearest_node_index(y, x, river_lat, river_long)
      self._river[index] = geodesic((river_lat[nearest_node_index], river_long[nearest_node_index]), (y, x)).m
      dist_river = self._river[index]
    if dist_end == 0:
      self._end[index] = nx.astar_path_length(G, self._state, dest)
      dist_end = self._end[index]
    #避難先に到着（エピソード終了）
    if self._state == dest:
      reward = 50
      #reachでエピソード終了を判定
      reach = 1
    #ひとつ前の状態より避難先に近づいていたら報酬
    elif dist_end < self._dist_end:
      #避難開始地点より遠ざかっていたら報酬0
      if self._a < dist_end:
        reward = 0
      #近づいていたら報酬
      else:
        reward = ((- dist_end + self._a) / self._a) / 10
      reach = 0
    #ひとつ前の状態より避難先から遠ざかっていたら報酬0
    else:
      reward = 0
      reach = 0
    #閾値より小さかったら罰を与える
    if dist_river < self._river_mean:
      reward -= 10 / dist_river
    if height < self._height_mean:
      reward -= 0.1 / height
    if dist_landslide < self._landslide_mean:
      reward -= 10 / dist_landslide
    #self._height，self._dist_river，self._dist_landslide，self._dist_endを更新
    self._height = height
    self._dist_river = dist_river
    self._dist_landslide = dist_landslide
    self._dist_end = dist_end
    return self._state, reward, reach
#Q値の設定
class MyQTable():
  def __init__(self, u, v, Q):
    self._Q = [0] * len(Q)
    self._u = u
    self._v = v
  #行動価値関数Q値の更新
  def update_Qtable(self, index, reward, next_state):#Q値更新
    gamma = 0.9
    alpha = 0.95
    #next_state中でself._uと一致するインデックスを全て抜き出す（現在の状態から遷移できる状態を抜き出す）
    temp_index = my_index_multi(self._u, next_state)
    #移動先の状態のQ値の一つをnext_maxQに仮代入
    next_maxQ = self._Q[temp_index[0]]
    #移動先の状態で一番行動価値が高いものを next_maxQに代入
    for x in temp_index:
      if next_maxQ < self._Q[x]:
          next_maxQ = self._Q[x]
    #行動価値関数Q値の更新
    self._Q[index] = (1 - alpha) * self._Q[index] + alpha * (reward + gamma * next_maxQ)
    return self._Q
  #soft-max行動選択
  def softmax_selection(self, state):
    beta = 7
    values = []
    index = []
    #現在の状態の状態価値関数のインデックス抜き出す
    temp_index = my_index_multi(self._u, state)
    #現在の状態でとれる行動のQ値，インデックスをlist化
    for x in temp_index:
        values.append(self._Q[x])
        index.append(x)
    # softmax選択の分母の計算
    sum_exp_values = sum([np.exp(v*beta) for v in values])
    # 確率分布の生成
    p = [np.exp(v*beta)/sum_exp_values for v in values]
    # 確率分布pに従ってランダムで選択
    action = np.random.choice(np.arange(len(values)), p=p)
    return index[action]

def main():
  #shpをdataframe化したものをpklで保存している
  gdf_road_iizuka_node = pd.read_pickle('Iizuka/gdf_road_iizuka_node.pkl') #道路のnode（point）
  gdf_place_Iizuka = pd.read_pickle('Iizuka/gdf_place_Iizuka.pkl') #避難所（point）
  gdf_road_iizuka_edge_for_path = pd.read_pickle('Iizuka/gdf_road_iizuka_edge_for_path.pkl') #edgeのuとvを入れ替えたものをedgeに追加して重複したものを消したgdf（line）
  gdf_river_iizuka_points = pd.read_pickle('Iizuka/gdf_river_iizuka_points.pkl') #河川の流路をポイント化したgdf（point）
  gdf_landslide_points = pd.read_pickle('Iizuka/gdf_landslide_points.pkl') #土砂災害危険箇所（polygon）の重心を取ったgdf（point）
  gdf_area_points = pd.read_pickle('Iizuka/gdf_area_points.pkl') #浸水予想地域の外縁をポイント化したgdf（point）
  f = open("./height.txt","rb")
  l_height = pickle.load(f) #nodeと対応づけて標高を格納したlist
  df_height = pd.read_pickle('Iizuka/df_height.pkl') #標高（point）
  #edgeをlist化
  u = gdf_road_iizuka_edge_for_path['u'].values.tolist()
  v = gdf_road_iizuka_edge_for_path['v'].values.tolist()
  #nodeをlist化
  osmid = gdf_road_iizuka_node['osmid'].values.tolist()
  node_lat = gdf_road_iizuka_node['geometry'].y.values.tolist()
  node_long = gdf_road_iizuka_node['geometry'].x.values.tolist()
  #河川の流路のポイントをリスト化
  river_lat = gdf_river_iizuka_points['geometry'].y.values.tolist()
  river_long = gdf_river_iizuka_points['geometry'].x.values.tolist()
  #土砂災害危険箇所の重心をリスト化
  landslide_lat = gdf_landslide_points['geometry'].y.values.tolist()
  landslide_long = gdf_landslide_points['geometry'].x.values.tolist()
  #評価の基準の3つの値を計算
  height_mean = gdf_area_points['height'].mean()
  dist_river_mean = gdf_area_points['dist_river'].mean()
  dist_landslide_mean = gdf_place_Iizuka['dist_landslide'].mean()
  #道路情報をNetworkXでネットワーク化
  DG = gdf_to_network(gdf_road_iizuka_edge_for_path, gdf_road_iizuka_node)
  #避難開始地点（緯度・経度）をランダムに決める
  start_lat = random.uniform(33.61, 33.685)
  start_long = random.uniform(130.645, 130.745)
  #出発地のノードのインデックスを取得
  start_index = find_nearest_node_index(start_lat, start_long, node_lat, node_long)
  #出発地のosmidを取得
  start = gdf_road_iizuka_node.at[start_index, 'osmid']
  #出発地から近い避難所20ヶ所に最も近い交差点のosmidと避難所のインデックスをリストで取得
  l_end, index_end = find_nearest_shelter(start_lat, start_long, gdf_place_Iizuka, 20, node_lat, node_long, osmid)
  score_max = 0
  shelter_count = 0
  for x in range(len(index_end)):
    gdf_astar_path_edge, _, _ = astar(DG, start, l_end[x], gdf_road_iizuka_edge_for_path, gdf_road_iizuka_node)
    #橋を通っているかどうか判定
    if 'yes' not in gdf_astar_path_edge['bridge'].tolist():
      shelter_count += 1
      #避難所のスコアを比較して最も大きいものに一番近い交差点のosmidをendに代入
      if score_max < gdf_place_Iizuka.at[index_end[x], 'score']:
        score_max = gdf_place_Iizuka.at[index_end[x], 'score']
        end = l_end[x]
      #橋を通っていない避難所について5ヶ所のスコアを参照したらbreak
      if shelter_count == 5:
        break
  #最短経路のedge，node，weightを取得
  gdf_astar_path_edge, gdf_astar_path_node, astar_path_weight = astar(DG, start, end, gdf_road_iizuka_edge_for_path, gdf_road_iizuka_node)

  num_episodes = 5000 #総エピソード回数
  max_number_of_steps = 200 #各エピソードの行動数（ステップ数）
  l_optimal_path_edge = []
  l_path = []
  num_routes = 5 #強化学習を5回行って5つの経路を算出する
  #強化学習で用いる各リストを初期化
  Q = [0] * len(u)
  dist_end = [0] * len(osmid)
  dist_river = [0] * len(osmid)
  dist_landslide = [0] * len(osmid)
  #強化学習
  for x in range(num_routes): #強化学習を5回行う
    env = MyEnvironmentSimulator(u, v, start, node_lat, node_long, osmid, dist_end, dist_river, dist_landslide, x, astar_path_weight, dist_river_mean, dist_landslide_mean, height_mean)
    tab = MyQTable(u, v, Q)
    #学習が収束したかどうか確認する諸変数
    itterarion = 0
    pre_step = 0
    itterated_step = 0
    for episode in range(num_episodes):  #エピソード回数分繰り返す
      start_time = time.time()
      #環境のリセット
      state = env.reset(start)
      episode_reward = 0
      optimal_path = [start]
      for t in range(max_number_of_steps):  #各エピソードで行う行動数分繰り返す
        index = tab.softmax_selection(state)  #soft-max選択，indexはnext_stateのedge上でのインデックス
        next_state, reward, reach = env.step(index, end, DG, river_lat,river_long ,l_height, landslide_lat,landslide_long) #行動による状態変化，index_nodeはnext_stateのnode上のインデックス
        episode_reward += reward
        step = t 
        q_table = tab.update_Qtable(index, reward, next_state)#Q値の更新
        state = next_state #状態を遷移
        optimal_path.append(next_state) #経路を記録
        #避難先にたどり着いていたらbreak
        if reach == 1:
          break
      elapsed_time = time.time() - start_time
      #ひとつ前のエピソードのステップ数と変わっていたら繰り返し数リセット
      if itterated_step != step:
        itterarion = 0
      #ひとつ前のエピソードのステップ数と同じで最大ステップ数でなかったら中に入る
      if step == pre_step and step != max_number_of_steps - 1:
        #繰り返されているステップと同じステップだったら繰り返し数プラス
        if itterated_step == step:
          itterarion = 1 + itterarion
        #繰り返されているステップと違うステップだったら繰り返し数リセット
        else:
          itterarion = 0
        #繰り返されているステップ数を更新
        itterated_step = step
      #20回同じステップ数で避難先に辿り着いていたらbreak
      if itterarion == 20 and reach == 1:
        break
      #pre_stepを更新
      pre_step = step
      print(f'Episode:{episode:4.0f}, Step:{t:3.0f}, episord_reward:{episode_reward:.2f}, elapsed_time:{elapsed_time:.4f}[sec], route{x}')
    #算出した経路のosmidのリストからgdf_edgeとgdf_nodeを取得し，リスト化
    gdf_path_edge, gdf_path_node = l_to_gdf(optimal_path, gdf_road_iizuka_edge_for_path, gdf_road_iizuka_node)
    l_path.append(gdf_path_node)
    l_optimal_path_edge.append(gdf_path_edge)
  #経路の諸値を格納するdataframe
  df_optimal = pd.DataFrame()
  df_shortest = pd.DataFrame()
  max_score = 3 #安全度の上限値
  #強化学習で算出した経路の評価
  for x in range(len(l_path)):
    points = edge_to_points(l_optimal_path_edge[x]) #経路上に評価点を設置
    points = eval(points, river_lat, river_long, landslide_lat, landslide_long, df_height)  #評価点についてdist_river，dist_landslide，heightを算出
    #評価点についてcos_simを計算してl_cosに格納
    l_vector = []
    for y in range(len(points)):
        l_vector.append([points.at[y, 'geometry'].y, points.at[y, 'geometry'].x])
    l_cos = []
    for y in range(len(l_vector)-2):
      l_cos.append(cos_sim(np.array(l_vector[y+1])-np.array(l_vector[y]) ,np.array(l_vector[y+2])-np.array(l_vector[y+1])))
    #スコアを算出
    river_score = points['dist_river'].sum() / len(points) / dist_river_mean if points['dist_river'].sum() / len(points) / dist_river_mean < max_score else max_score
    landslide_score = points['dist_landslide'].sum() / len(points) / dist_landslide_mean if points['dist_landslide'].sum() / len(points) / dist_landslide_mean < max_score else max_score
    height_score = points['height'].sum() / len(points) / height_mean if points['height'].sum() / len(points) / height_mean < max_score else max_score
    length_score = astar_path_weight / l_optimal_path_edge[x]['length'].sum()
    cos_score = 1 - len([z for z in l_cos if z < 0.5]) * 0.1
    #諸値をdataframe化
    df_optimal = df_optimal.append({'Optimal route length': l_optimal_path_edge[x]['length'].sum(),
                                    'steps': len(l_path[x]),
                                    'river_score': river_score,
                                    'landslide_score': landslide_score,
                                    'height_score': height_score,
                                    'length_score': length_score,
                                    'score': (river_score + landslide_score + height_score) / 3,
                                    'cos sim': cos_score,
                                    'rl_score': (river_score + landslide_score + height_score + length_score + cos_score) / 5},
                                    ignore_index = True)
  #最短経路の評価
  points = edge_to_points(gdf_astar_path_edge) #経路上に評価点を設置
  points = eval(points, river_lat, river_long, landslide_lat, landslide_long, df_height) #評価点についてdist_river，dist_landslide，heightを算出
  #評価点についてcos_simを計算してl_cosに格納
  l_vector = []
  for x in range(len(points)):
    l_vector.append([points.at[x, 'geometry'].y, points.at[x, 'geometry'].x])
  l_cos = []
  for x in range(len(l_vector)-2):
    l_cos.append(cos_sim(np.array(l_vector[x+1])-np.array(l_vector[x]) ,np.array(l_vector[x+2])-np.array(l_vector[x+1])))
  #スコアを算出
  river_score = points['dist_river'].sum() / len(points) / dist_river_mean if points['dist_river'].sum() / len(points) / dist_river_mean < max_score else max_score
  landslide_score = points['dist_landslide'].sum() / len(points) / dist_landslide_mean if points['dist_landslide'].sum() / len(points) / dist_landslide_mean < max_score else max_score
  height_score = points['height'].sum() / len(points) / height_mean if points['height'].sum() / len(points) / height_mean < max_score else max_score
  #諸値をdataframe化
  df_shortest = df_shortest.append({'shortest route length': astar_path_weight,
                                    'steps': len(gdf_astar_path_node),
                                    'start': gdf_astar_path_node.at[0, 'osmid'],
                                    'end': gdf_astar_path_node.at[len(gdf_astar_path_node) - 1, 'osmid'],
                                    'river_score': river_score,
                                    'landslide_score': landslide_score,
                                    'height_score': height_score,
                                    'score': (river_score + landslide_score + height_score) / 3},
                                    ignore_index = True)
  #経路の諸値をcsv化
  df_optimal.to_csv('dataframe_optimal.csv')
  df_shortest.to_csv('dataframe_shortest.csv')

if __name__ == '__main__':
  main()
