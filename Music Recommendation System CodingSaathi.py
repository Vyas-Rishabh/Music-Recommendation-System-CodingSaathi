#!/usr/bin/env python
# coding: utf-8

# # Music Recommendation System

# #### Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import warnings
warnings.filterwarnings('ignore')


# ## Reading the dataset

# In[2]:


members = pd.read_csv("members.csv")
members


# In[3]:


songs=pd.read_csv("songs.csv", nrows=20000)
songs


# In[4]:


songs_info=pd.read_csv("song_extra_info.csv")
songs_info


# In[5]:


submission=pd.read_csv("sample_submission.csv", nrows=20000)
submission


# In[6]:


train_data=pd.read_csv("train.csv", nrows=20000)
train_data


# In[7]:


test_data=pd.read_csv("test.csv", nrows=20000)
test_data


# In[8]:


print(f"The songs_data has {songs.shape[0]} rows and {songs.shape[1]} columns")
print(f"The songs_extra_info_data has {songs_info.shape[0]} rows and {songs_info.shape[1]} columns")
print(f"The members_data has {members.shape[0]} rows and {members.shape[1]} columns")
print(f"The sample_submission_data has {submission.shape[0]} rows and {submission.shape[1]} columns")
print(f"The train_data has {train_data.shape[0]} rows and {train_data.shape[1]} columns")
print(f"The test_data has {test_data.shape[0]} rows and {test_data.shape[1]} columns")


# In[9]:


songs.describe()


# In[10]:


print("Columns present in the songs data are:")
for columns in songs.columns:
    print(columns)


# In[11]:


print(f"Number of records : {songs.shape[0]}")
print(f"Count of distinct song lengths : {len(songs.song_length.unique())}")
print(f"Count of distinct genre ids : {len(songs.genre_ids.unique())}")
print(f"Count of distinct artist name : {len(songs.artist_name.unique())}")
print(f"Count of distinct composer : {len(songs.composer.unique())}")
print(f"Count of distinct lyricist : {len(songs.lyricist.unique())}")
print(f"Count of distinct language : {len(songs.language.unique())}")


# ## Data preprocessing

# In[12]:


plt.figure(figsize= (12, 5))
sns.set_style("darkgrid")
ax = sns.countplot(x = songs.language, data = songs.language, palette="deep")
ax.set_title("Countplot for Languages")
plt.show()


# In[13]:


print("Columns present in the Members Data are:")
for columns in members.columns:
    print(columns)


# In[14]:


plt.figure(figsize= (7, 5))
sns.set_style("darkgrid")
sns.countplot(x='gender', data=members, palette="muted")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.title("Count plot for Gender")


# In[15]:


plt.figure(figsize= (7 ,5))
sns.countplot(x="registered_via", data=members, palette="muted")
plt.xlabel("Registration Method")
plt.ylabel("Count")
plt.title("Count plot for Registration Method")


# In[16]:


print(f"Total number of records : {train_data.shape[0]}")


# In[17]:


plt.figure(figsize= (7, 5))
sns.countplot(x='target', data=train_data, palette='brg')
plt.xlabel("Target")
plt.ylabel("Count")
plt.title("Count plot for System Tab there are missing")


# In[18]:


print("Total percentage for NaN value in target column : ", (train_data["target"].isna().sum()/len(train_data["target"]))*100,"%")


# In[19]:


duplicate_value1 = len(train_data["song_id"])-train_data["song_id"].nunique()
print("Total number of duplicate song id : ", duplicate_value1)
print("Total percentage of duplicate song id : ", (duplicate_value1/len(train_data["song_id"]))*100,"%")


# In[20]:


plt.figure(figsize=(7, 5))
sns.countplot(y=train_data["target"], data=train_data, palette="Accent")
plt.ylabel("Target Classes")
plt.xlabel("Frequency ")
plt.show()


# In[21]:


songs_info.head()


# In[22]:


songs_info.isnull().sum()


# In[23]:


songs.isnull().sum()


# In[24]:


songs['genre_ids'].fillna(' ', inplace=True)


# In[25]:


songs['composer'].fillna(' ', inplace=True)
songs['lyricist'].fillna(' ', inplace=True)
songs['language'].fillna((52.0), inplace=True)


# In[26]:


songs.isnull().sum()


# In[27]:


train_data.isnull().sum()


# In[28]:


train_data = train_data.drop(['source_system_tab', 'source_screen_name', 'source_type'], axis=1)
train_data.head()


# In[29]:


train_data.shape


# In[30]:


train_data.rename(columns={'msno':"user_id"}, inplace=True)
train_data.head()


# In[31]:


songs.head()


# In[32]:


df = train_data.merge(songs, on="song_id")
df.head()


# In[33]:


df = df.drop(['song_length', 'language'], axis=1)
df.head()


# In[34]:


songs_info.head()


# In[35]:


df = df.merge(songs_info,on="song_id").drop('isrc',axis=1)
df.head()


# In[36]:


df.rename(columns={'name':'song_name'}, inplace=True)
df.head()


# ## Data cleaning

# In[37]:


df['genre_ids'].value_counts()


# In[38]:


df['genre_ids']=df['genre_ids'].str.replace('|', ' ', regex=True)
df['genre_ids'].value_counts()


# In[39]:


df['artist_name']=df['artist_name'].str.replace('|', ' ', regex=True)
df['composer']=df['composer'].str.replace('/', ' ', regex=True)
df['lyricist']=df['lyricist'].str.replace('/', ' ', regex=True)
df['artist_name']=df['artist_name'].str.lower()
df['composer']=df['composer'].str.lower()
df['lyricist']=df['lyricist'].str.lower()


# In[42]:


df['songs_details']=df['artist_name']+' '+df['composer']+df['lyricist']
df.head()


# In[43]:


df.user_id.value_counts()


# In[45]:


df.duplicated().sum()


# In[46]:


#Creating a copy file before performing a similarity
main_df=df.copy()
main_df.head()


# In[47]:


main_df.songs_details.duplicated().sum()


# In[49]:


main_df.shape


# In[50]:


main_df.duplicated().sum()


# In[51]:


main_df=main_df.drop(['user_id'], axis=1)


# In[52]:


main_df


# In[53]:


main_df.duplicated().sum()


# In[54]:


main_df=main_df.drop_duplicates()
main_df


# In[55]:


main_df.reset_index(inplace=True)


# In[57]:


main_df.shape


# ## Mapping frequent words

# In[59]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(analyzer='word', stop_words='english')
tfidf_matrix=tfidf.fit_transform(main_df['songs_details'])


# In[60]:


tfidf_matrix


# ## Building Similarity

# In[61]:


from sklearn.metrics.pairwise import cosine_similarity


# In[62]:


cosine_similarity = cosine_similarity(tfidf_matrix)


# In[63]:


cosine_similarity


# In[64]:


sorted(list(enumerate(cosine_similarity[0])), reverse=True, key=lambda x:x[1])[1:6]


# In[65]:


#In which you can recommend only index
def recommend(song):
    song_index=main_df[main_df['song_name']==song].index[0]
    distances=cosine_similarity[song_index]
    song_list=sorted(list(enumerate(cosine_similarity[0])), reverse=True, key=lambda x:x[1])[1:6]
    for i in song_list:
        print(i[0])


# ## User based Recommender - Content

# In[67]:


def recommend(song):
    song_index=main_df[main_df['song_name']==song].index[0]
    distances=cosine_similarity[song_index]
    song_list=sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:10]
    for i in song_list:
        print(main_df.iloc[i[0]].song_name)


# In[68]:


recommend('Panda')


# # Results:
# 
# The results of the Music Recommendation System project are highly dependent on the specific implementation, data preprocessing
# techniques, and model selection. By employing collaborative filtering or content-based filtering approaches, the recommendation system was able to provide personalized music recommendations to users. The system's recommendations aimed to enhance user engagement, satisfaction, and enjoyment of the music streaming platform.

# You can find this project on <a href="https://github.com/Vyas-Rishabh/Music-Recommendation-System-CodingSaathi"><b>GitHub.</b></a>
