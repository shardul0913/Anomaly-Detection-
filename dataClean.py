import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pyod.models.auto_encoder import AutoEncoder

x = pd.read_csv("train.csv")

df = x.loc[x['prop_id'] == 104517]
df = df.loc[df['srch_room_count'] == 1]
df = df.loc[df['visitor_location_country_id'] == 219]
df = df[['date_time', 'price_usd', 'srch_booking_window', 'srch_saturday_night_bool','prop_review_score','prop_brand_bool','prop_location_score1',
      'srch_booking_window','orig_destination_distance']]

df.to_excel('ViewData.xlsx')
print(df)

cleanDf = pd.read_excel('ViewData.xlsx')
print(cleanDf['date_time'])
cleanDf.info()
cleanDf.dropna(inplace=True)
cleanDf['date_time'] = pd.to_datetime(cleanDf['date_time'])

# cleanDf.set_index('date_time', inplace=True)
# cleanDf.plot(figsize=(20,10))

####### plot the time series data ######

cleanDf.plot(x='date_time', y='price_usd', figsize=(12,6))
plt.show()

###### The distribution on saturday vs non saturday

sat = cleanDf.loc[cleanDf['srch_saturday_night_bool'] == 1,'price_usd']
nonSat = cleanDf.loc[cleanDf['srch_saturday_night_bool'] == 0,'price_usd']
print(sat)
print(nonSat)
sns.distplot(sat,bins=50)
sns.distplot(nonSat,bins=50)
plt.show()

###### K means to detect the anomaly in the dataset ######

dropClean = cleanDf.drop(['date_time'],axis=1)
score = []
for i in range(1,15):
    kFit = KMeans(n_clusters=i).fit(dropClean)
    score.append(kFit.score(dropClean))
sns.lineplot(range(1,15),score)
plt.show()

###### choosing the cluster value of 8 ######

kFit = KMeans(n_clusters=8).fit(dropClean)
kFit.predict(dropClean)
label = kFit.labels_

fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
ax.scatter(dropClean.iloc[:,0], dropClean.iloc[:,1], dropClean.iloc[:,2],
          c=label)
ax.set_xlabel("price_usd")
ax.set_ylabel("srch_booking_window")
ax.set_zlabel("srch_saturday_night_bool")
plt.title("K Means", fontsize=14)
plt.show()

###### PCA #######

### dropClean = StandardScaler().fit_transform(dropClean)


dropClean_std = StandardScaler().fit_transform(dropClean)
mean_vec = np.mean(dropClean_std, axis=0)
##Covarinace Matrix
cov_mat = (dropClean_std - mean_vec).T.dot((dropClean_std - mean_vec)) / (dropClean_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)
##Eigen Values of Covariance Matrix
eigVals, eigVecs = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' %eigVecs )
print('\nEigenvalues \n%s' %eigVals)

eigens = {}
for i in range(len(eigVals)):
    eigens[eigVals[i]] = eigVecs[:,i]
print(eigens)

##### Plot the PCA #####

total = sum(eigVals)
variance = [(i / total)*100 for i in sorted(eigVals, reverse=True)]
cummuVar = np.cumsum(variance)
plt.bar(range(len(variance)), variance, alpha=0.3, align='center', color = 'b')
plt.step(range(len(cummuVar)), cummuVar)
plt.show()
plt.ylabel('ratio variance')
plt.xlabel('Principal components')
#


pca = PCA(n_components=4)
dropCleanPCA = pca.fit_transform(dropClean)
# standardize these 2 new features
scaler = StandardScaler()
np_scaled = scaler.fit_transform(dropCleanPCA)
dataFinal = pd.DataFrame(np_scaled)

allDistance = pd.Series()
newK = KMeans(n_clusters=8).fit(dataFinal)

for i in range(len(dataFinal)):
    a = np.array(dataFinal.loc[i])
    # getting the clusters for the selected data row
    b = newK.cluster_centers_[newK.labels_[i]-1]
    ##the euclidian distance
    allDistance.set_value(i, np.linalg.norm(a - b))

#assumption of outlier percentage
threshfrac = 0.01
number_of_outliers = int(threshfrac*len(allDistance))
thresh = allDistance.nlargest(number_of_outliers).min()
dropClean['anomaly1'] = (allDistance >= thresh).astype(int)
print(dropClean)
dropClean.to_excel('CalculatdAnomaly.xlsx')

#######

cleanDf = cleanDf.sort_values('date_time',ascending=True)
dropClean = pd.read_excel('CalculatdAnomaly.xlsx')
# dropClean['Unnamed: 0'] = cleanDf['date_time']
fig, ax = plt.subplots(figsize=(10,6))
a = dropClean.loc[dropClean['anomaly1'] == 1, [dropClean.iloc[:,0], 'price_usd']]
ax.plot(dropClean.iloc[:,0], dropClean['price_usd'], color='blue', label='Normal')
ax.scatter(a.iloc[:,0],a['price_usd'], color='red', label='Anomaly')
plt.show()

########### Autoencoder ############

# x = pd.read_csv("train.csv")
# x.info()

# df = x.loc[x['srch_room_count'] == 1]
# df = df[df['prop_id'].isin([893,10404,21315,29604])]
# df = df.loc[df['visitor_location_country_id'] == 219]
# # df = df[(df['prop_id'] == 893 or df['prop_id'] == 10404 or df['prop_id'] == 21315 or df['prop_id'] == 29604)]
# df = df[['date_time','prop_id','price_usd','srch_id','site_id','visitor_location_country_id','srch_booking_window',
#          'srch_saturday_night_bool','visitor_hist_starrating','visitor_hist_adr_usd','prop_country_id','prop_review_score',
#          'prop_brand_bool','prop_starrating','prop_location_score1','srch_booking_window','orig_destination_distance']]
# df.fillna(method ='bfill')
# df.to_excel('nnViewData.xlsx')
# print(df)

# x = pd.read_csv("test.csv")
# x.info()
# df = x.loc[x['srch_room_count'] == 1]
# df = df[df['prop_id'].isin([893,10404,21315,29604])]
# df = df.loc[df['visitor_location_country_id'] == 219]
# # df = df[(df['prop_id'] == 893 or df['prop_id'] == 10404 or df['prop_id'] == 21315 or df['prop_id'] == 29604)]
# df = df[['date_time','prop_id','price_usd','srch_id','site_id','visitor_location_country_id','srch_booking_window',
#          'srch_saturday_night_bool','visitor_hist_starrating','visitor_hist_adr_usd','prop_country_id','prop_review_score',
#          'prop_brand_bool','prop_starrating','prop_location_score1','srch_booking_window','orig_destination_distance']]
# df.fillna(0)
# df.to_excel('nnViewDataTest.xlsx')
# print(df)

nnData = pd.read_excel("nnViewData.xlsx")
nnData = nnData.drop(['date_time'],axis=1)
dropCleanScale = StandardScaler().fit_transform(nnData)
dropCleanScale = pd.DataFrame(dropCleanScale)

nnDataTest = pd.read_excel("nnViewDataTest.xlsx")
nnDataTest = nnDataTest.drop(['date_time'],axis=1)
dropCleanScaleTest = StandardScaler().fit_transform(nnDataTest)
dropCleanScaleTest = pd.DataFrame(dropCleanScaleTest)

clf1 = AutoEncoder(hidden_neurons =[14, 2, 2, 14])
clf1.fit(dropCleanScale)
y_train_scores1 = clf1.decision_scores_

clf2 = AutoEncoder(hidden_neurons =[14, 10, 2, 10, 14])
clf2.fit(dropCleanScale)
y_train_scores2 = clf2.decision_scores_

y_test1 = clf1.decision_function(dropCleanScaleTest)
y_test2 = clf2.decision_function(dropCleanScaleTest)
## plotting the Remaining lifetime score

plt.hist(y_test1, bins='auto',color='green')
plt.hist(y_test2, bins='auto',color='blue')

plt.title("Histogram for Model Clf1 Anomaly Scores")
plt.show()

df_test = y_train_scores2.copy()
df_test['score'] = y_test2
df_test['cluster'] = np.where(df_test['score']<4, 0, 1)
df_test['cluster'].value_counts()
df_test.groupby('cluster').mean()