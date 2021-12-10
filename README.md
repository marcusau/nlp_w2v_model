# word2vec model 有兩個.. 1個light version size不足200 m ...但準確度合理.... 若一個是heavy version因為size逾500m...因此. ,現在透過skype發給你們. .因為wechat不容許size >500 傳送


首先必須強調, 訓練及deployment所使用的open-library是不同的.

train: openlibrary 使用gensim 因為其可以在no GPU環境下進行大量語料training 而且速度很快.. 即使有幾十萬row的data..也可以半小時之內在cpu環境下完成訓練.....
github內的model 已有zip version


gensim 是專門訓練word2vec model的高速open -library....

即使在你們Local機上, 只有data齊全. 一樣可以輕易進行training

半小時即可..當然.在training 期間.. Local machine may be slowed down a bit...但你可以在訓練之前......調教使用多少個cpu core來訓練. 但首先你們要有google drive account請先登記


jupyter notebook內. 5.2 部份. 

workers= no. of cpu cores to be used for training the model

如果把workers數目減少. .可減輕local machine 的負擔.. 但同時, 會減慢training 速度...這個你們需要衡量

我在google colab進行training則沒有這方面的擔心.. 可全力使用所有colab上的cpu cores進行訓練

這便是因為OOV問題... 而要把訓練model 及deploy model分別使用兩個不同的open library的理由.
train :gensim 
model conversion:  python -m pymagnitude.converter -i <PATH TO FILE TO BE CONVERTED> -o <OUTPUT PATH FOR MAGNITUDE FILE>
deployment : pymagnitude 



deployment則要另一個open library : https://github.com/plasticityai/magnitude

因為gensim 有一個缺點: gensim 所訓練出來的word2vec model. input一定是文字或中文詞語, 即string or list of string . 問題是, 如果該中文詞語是訓練時沒有出現的新詞語. 如近年提及的NFT或元宇宙等..則Gensim model會出現out-of-vocabulary (OOV)的情況, 使Model無法對新詞進行 vectorization

因此. 使用 https://github.com/plasticityai/magnitude 這個open-library就是針對OOV這個問題, 而採用的Deployment open -library

只要按照pymagnitude documentation內的指示. 把gensim 所訓練出來的Model 轉換成pymagnitude所採用的格式.  然後在deployment script 採用 轉換後的model ..便可... 

在我的個案 我便採用這句command :   python -m pymagnitude.converter -i  word2vec_training\model\gensim\etnet_w2v.bin   -o  C:\Users\marcus\PycharmProjects\word2vec_training\model\magnitude\boc_app_heavy.magnitude 

而如何透過pymagnitude 這個Open library 而deployment word2vec model (boc_app_heavy.magnitude )   請看deployment.py script





現在可以開始解釋.  如何training一個word2vec model.

如果local machine的內存足夠, 應盡量使用更多的語料...語料越多, 訓練出來的Model越準

我在github內已把三個資料來源,進行分詞及處理, 包括:
1. THUCNews (這是內地許多大專用來訓練或測試NLP程序專用的新聞語料, 我只集中選取了財經新聞、遊戲(因為騰訊是網上遊戲公司).  以及時尚的新聞(因為經濟通Lifestyle 有部份內容是針對時尚新聞)

2. 經濟通lifestyle內的針對財經新聞的文章

3. 經濟通2018-2020的製造的財經新聞

如果句子內同時出現中文及英文, 也可以空隔作分隔, 但需要留意, 英文詞語是否專有名詞. 例如,  Donald Trump 等,, 應該把兩個英文字合併成一個單詞 DonaldTrump處理

有部份專有名詞是否採用空格分開, 或是合拼處理.... 可能需要人手作鑑定







word2vec model training的input 只可以是 list of string or string ...

因此, 不能把未經過處理的中文句子或段落直接, 用作訓練之用。

以中文文章為例. 所以用來訓練的Input data ,必須經過三步處理:
1. 把文章分段成為句子, 可以句號或其他分割完整的句子的標點符號, 作為seperator
2. 把句子進行分詞(word segementation或 tokenization)  . 使句子變成 以空格分開的中文詞語.  建議使用結巴作為分詞工具..
3. 在完成分詞後, 把標點符號從文檔中移除, 包括, 包號, 逗號, 分號、感嘆號等...只留下文字部份.

在完成三個step後.  training dataset應該變成這個模樣:

文檔中每一行為一句.

每句之內, 每個詞語以空隔分開.

我在製造語料數據時, 也花了好幾天, 用內眼去鑑定部份語料,  不可能做到100%準確, 但可提升詞語之間的準繩度

在完成語料準備後,    可在python script 內, input txt file 然後以'\n' 把句子分別, 再以空格分開每一句句子內的詞語, 結果造成list of string list


我在jupyner notebook內已有用作input txt file的function


然後把 幾個 list of string list 合併在一起, 成為一個大集合. 這便是model input


### Model training
只要text input 成為list of string list 以及調教完成hyperparaemters 即可使用5.2-5.3進行訓練.   以github內已完成處理的語料, 在local machine 大概需時半小時. 

然後按照jupyter notebook 第五部份.. 自行調校訓練model 所需的hyper-parameter
https://blog.csdn.net/szlcw1/article/details/52751314

我試過把size調升至500  發覺分別不大

然後在5.2-5.3 部份便是已寫好的model training 主要部份

5.5 部份便是把完成訓練的Model 存放在local machine 內指定位置  完成訓練的word2vec model....最主要的是.etnet_w2v.bin   因為bin file size較細... 使用時速度較良好


在完成訓練gensim word2vec model後. 建議不要馬上儲存. 應該在jupyter notebook 內, 對Model進行簡單測試... 如下圖

如上圖指,  輸入「陳志全」. 看看AI Model能否辨別出, 哪幾個詞語跟「陳志全」關係最密切, 結果出來以下字眼

如果output字眼, 跟input字眼關係不大密切, 即model 效果欠佳. 建議先更改hyperparameter的設定,, 例如提升iter 次數, 或 減少window_size



### Model type conversion

但最重要... 即使model 效果良好, 但不要馬上在deployment上使用, 
因為gensim word2vec model有嚴重的OOV 問題... 即不能對部份新詞作出處理......強行使用會出現Error 情況, 因此必須把gensim model convert成為magnitude 模式..... 
然後使用pymagnitude-light 這個openlibrary進行deployment

你們會發現, 這裏有兩個pymagnitutde 的openlibrary
1. pymagnitude (https://github.com/plasticityai/magnitude)
2. pymagnitutde -light (https://github.com/davebulaval/magnitude-light)


現在解釋為何需要兩個不同的pymagnitude 

因為原始版本的pymagnitude (https://github.com/plasticityai/magnitude)  功能最齊全....所以應使用它作為工具, 把gensim word2vec model 轉換成 magnitude format.

但原始版本的pymagnitude 是重量級open-library 在deployment運作速度較慢. 因此, 本人只採用原始版本的pymagnitude 把gensim word2vec model 轉換為magnitude format 

然後採用size較細, deployment 速度快得多的pymagnitutde -light (https://github.com/davebulaval/magnitude-light) 進行 model deployment


model轉換的方法及command line 剛才已提及過.. 


#### deployment :
deployment 請自行看github 內的deployment.py 




