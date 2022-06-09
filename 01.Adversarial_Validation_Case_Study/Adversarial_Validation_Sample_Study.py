# ==============import packages=============
import pandas as pd
import numpy as np
from tqdm import tqdm
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

# Memory management
import gc
gc.enable()

# Plot
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# ==========数据抽样=========================
dtypes = {
    'MachineIdentifier':                                    'category',
    'ProductName':                                          'category',
    'EngineVersion':                                        'category',
    'AppVersion':                                           'category',
    'AvSigVersion':                                         'category',
    'IsBeta':                                               'int8',
    'RtpStateBitfield':                                     'float16',
    'IsSxsPassiveMode':                                     'int8',
    'DefaultBrowsersIdentifier':                            'float32',
    'AVProductStatesIdentifier':                            'float32',
    'AVProductsInstalled':                                  'float16',
    'AVProductsEnabled':                                    'float16',
    'HasTpm':                                               'int8',
    'CountryIdentifier':                                    'int16',
    'CityIdentifier':                                       'float32',
    'OrganizationIdentifier':                               'float16',
    'GeoNameIdentifier':                                    'float16',
    'LocaleEnglishNameIdentifier':                          'int16',
    'Platform':                                             'category',
    'Processor':                                            'category',
    'OsVer':                                                'category',
    'OsBuild':                                              'int16',
    'OsSuite':                                              'int16',
    'OsPlatformSubRelease':                                 'category',
    'OsBuildLab':                                           'category',
    'SkuEdition':                                           'category',
    'IsProtected':                                          'float16',
    'AutoSampleOptIn':                                      'int8',
    'PuaMode':                                              'category',
    'SMode':                                                'float16',
    'IeVerIdentifier':                                      'float16',
    'SmartScreen':                                          'category',
    'Firewall':                                             'float16',
    'UacLuaenable':                                         'float64',  # was 'float32'
    'Census_MDC2FormFactor':                                'category',
    'Census_DeviceFamily':                                  'category',
    'Census_OEMNameIdentifier':                             'float32',  # was 'float16'
    'Census_OEMModelIdentifier':                            'float32',
    'Census_ProcessorCoreCount':                            'float16',
    'Census_ProcessorManufacturerIdentifier':               'float16',
    'Census_ProcessorModelIdentifier':                      'float32',  # was 'float16'
    'Census_ProcessorClass':                                'category',
    'Census_PrimaryDiskTotalCapacity':                      'float64',  # was 'float32'
    'Census_PrimaryDiskTypeName':                           'category',
    'Census_SystemVolumeTotalCapacity':                     'float64',  # was 'float32'
    'Census_HasOpticalDiskDrive':                           'int8',
    'Census_TotalPhysicalRAM':                              'float32',
    'Census_ChassisTypeName':                               'category',
    'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float32',  # was 'float16'
    'Census_InternalPrimaryDisplayResolutionHorizontal':    'float32',  # was 'float16'
    'Census_InternalPrimaryDisplayResolutionVertical':      'float32',  # was 'float16'
    'Census_PowerPlatformRoleName':                         'category',
    'Census_InternalBatteryType':                           'category',
    'Census_InternalBatteryNumberOfCharges':                'float64',  # was 'float32'
    'Census_OSVersion':                                     'category',
    'Census_OSArchitecture':                                'category',
    'Census_OSBranch':                                      'category',
    'Census_OSBuildNumber':                                 'int16',
    'Census_OSBuildRevision':                               'int32',
    'Census_OSEdition':                                     'category',
    'Census_OSSkuName':                                     'category',
    'Census_OSInstallTypeName':                             'category',
    'Census_OSInstallLanguageIdentifier':                   'float16',
    'Census_OSUILocaleIdentifier':                          'int16',
    'Census_OSWUAutoUpdateOptionsName':                     'category',
    'Census_IsPortableOperatingSystem':                     'int8',
    'Census_GenuineStateName':                              'category',
    'Census_ActivationChannel':                             'category',
    'Census_IsFlightingInternal':                           'float16',
    'Census_IsFlightsDisabled':                             'float16',
    'Census_FlightRing':                                    'category',
    'Census_ThresholdOptIn':                                'float16',
    'Census_FirmwareManufacturerIdentifier':                'float16',
    'Census_FirmwareVersionIdentifier':                     'float32',
    'Census_IsSecureBootEnabled':                           'int8',
    'Census_IsWIMBootEnabled':                              'float16',
    'Census_IsVirtualDevice':                               'float16',
    'Census_IsTouchEnabled':                                'int8',
    'Census_IsPenCapable':                                  'int8',
    'Census_IsAlwaysOnAlwaysConnectedCapable':              'float16',
    'Wdft_IsGamer':                                         'float16',
    'Wdft_RegionIdentifier':                                'float16',
    'HasDetections':                                        'int8'
}

# # df_all = pd.read_csv('./input/train.csv.zip', dtype=dtypes) 
# df_all = pd.read_csv('/Users/amyqian/Downloads/microsoft-malware-prediction/train.csv',dtype=dtypes) 
# # 对训练集随机抽取2%的样本
# df_all = df_all.sample(frac=0.02, random_state=123)
# df_all.to_csv('./input/train_sample.csv', index=False)
# ========数据预处理==================
# 因为这次比赛的训练集有800万样本，测试集有700万样本。为了方便演示，这里我仅随机抽取训练集中2%的样本，而且不使用测试集的数据。我们稍后将从训练集中拆分一个数据集，作为我们的测试集。
# 不使用这次比赛原本的测试集可以节省很多时间。因为测试集有700万的样本，每做一次预测会消耗大量时间。
# 读取训练集中随机抽取的2%的样本

df_all = pd.read_csv('./input/train_sample.csv') 
print(df_all.head())

# 本次比赛的数据中提供了电脑的 Windows Defender（Windows系统自带的杀毒软件）版本号，所以我们可以通过该本版号发布的时间，粗略的推测采集该样本的时间。
# 这里AvSigVersionTimestamps就是各个Windows Defender版本对应的发布时间。
# 通过和该数据匹配，我们生成了一个新的字段 -- Date（日期）。这个字段稍后会起作用。

# 读取Windows Defender版本对应的发布时间
datedict = np.load('./input/AvSigVersionTimestamps.npy')
datedict = datedict[()]
# 生成新的变量Date
df_all['Date'] = df_all['AvSigVersion'].map(datedict)

# MachineIdentifier是每台电脑的唯一识别号，对于模型的预测没有任何帮助，所以剔除。
df_all.drop(['MachineIdentifier'], axis=1, inplace=True) 
print(df_all.head())

# =============数据清理===================================
# 这里无意义变量的定义是：变量的某个值（可以是空值）的占比大于99%。
# 比如，如果所有样本的「系统版本」都是Win7，那么「系统版本」这个变量就没有意义。
# 所以，如果一个变量，99%以上的样本，都是一个值，那么这个变量接近于无意义。
bad_cols = []
for col in df_all.columns:
    rate_train = df_all[col].value_counts(normalize=True, dropna=False).values[0]
    # print(rate_train)
    if rate_train > 0.99:
        bad_cols.append(col)

df_all = df_all.drop(bad_cols, axis=1)

print('Data Shape: ', df_all.shape)
print(bad_cols)

# 这里是通过EDA(Exploratory Data Analysis)的方式，人工判断的变量类型。

# 总共将变量分为
# * 数值变量（true_numerical_columns）
# * 一般的分类变量（categorical_columns）
# * 类别非常多的分类变量（categorical_columns_high_car）：比如中国的城市（北京、上海、深圳、重庆等等等...）

# 如果你对这次比赛的细节感兴趣，可以再深入研究为什么这样判断。这里就不详细阐述原因了。
true_numerical_columns = [
    'Census_PrimaryDiskTotalCapacity', 'Census_SystemVolumeTotalCapacity',
    'Census_TotalPhysicalRAM', 'Census_InternalBatteryNumberOfCharges'
]

categorical_columns_high_car = [
    'Census_FirmwareVersionIdentifier', 'Census_OEMModelIdentifier',
    'AVProductStatesIdentifier', 'Census_FirmwareManufacturerIdentifier',
    'Census_InternalPrimaryDiagonalDisplaySizeInInches',
    'Census_InternalPrimaryDisplayResolutionHorizontal',
    'Census_InternalPrimaryDisplayResolutionVertical',
    'Census_OEMNameIdentifier', 'Census_ProcessorModelIdentifier',
    'CityIdentifier', 'DefaultBrowsersIdentifier', 'OsBuildLab'
]

categorical_columns = [
    c for c in df_all.columns
    if c not in (['HasDetections', 'Date'] + true_numerical_columns +
                 categorical_columns_high_car)
]
print(categorical_columns)

### 编码 -- Label Encoding 
# 因为将使用的模型是[LightGBM](https://lightgbm.readthedocs.io/en/latest/)，所以我们需要对分类变量做编码。

# 这里用的方法是[Label Encoding](http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.factorize.html)。
# 对于one-hot encoding、dummy enconding还是factorize都可以将定性特征转化为定量特征，三种方法各有各自最适用的场合，
# 要根据实际情况作出最合理的选择，以便在数据处理的过程中得到最理想的结果。
def factor_data(df, col):
    df_labeled, _ = df[col].factorize(sort=True)
    # MAKE SMALLEST LABEL 1, RESERVE 0
    df_labeled += 1
    # MAKE NAN LARGEST LABEL
    df_labeled = np.where(df_labeled==0, df_labeled.max()+1, df_labeled)
    df[col] = df_labeled

for col in tqdm(categorical_columns + categorical_columns_high_car):
    factor_data(df_all, col) 

# =================构造测试集=================

# 像刚才提到的，因为没有使用测试集的数据，所以我们需要从训练集中拆分出一个数据集，作为我们的测试集，用于评价我们评估模型的方式是否有效。

# 因为训练集和测试集是根据时间划分的，所以我们从训练集拆分的测试集，同样也根据时间划分。

# 这是为了尽量模拟真实的测试集。
# 将样本根据时间排序
df_all = df_all.sort_values('Date').reset_index(drop=True) 
df_all.drop(['Date'], axis=1, inplace=True)

# 将前80%的样本作为训练集，后20%的样本作为测试集
df_test = df_all.iloc[int(0.8*len(df_all)):, ]
df_train = df_all.iloc[:int(0.8*len(df_all)), ]

print(df_train)
print(df_test)
## =====================对抗验证（Adversarial Validatiion）======================
# 定义新的Y
df_train['Is_Test'] = 0
df_test['Is_Test'] = 1

# 将 Train 和 Test 合成一个数据集。HasDetections是数据本来的Y，所以剔除。
df_adv = pd.concat([df_train, df_test])

adv_data = lgb.Dataset(
    data=df_adv.drop('Is_Test', axis=1), label=df_adv.loc[:, 'Is_Test'])

# 定义模型参数
params = {
    'boosting_type': 'gbdt',
    'colsample_bytree': 1,
    'learning_rate': 0.1,
    'max_depth': 5,
    'min_child_samples': 100,
    'min_child_weight': 1,
    'min_split_gain': 0.0,
    'num_leaves': 20,
    'objective': 'binary',
    'random_state': 50,
    'subsample': 1.0,
    'subsample_freq': 0,
    'metric': 'auc',
    'num_threads': 8
}

# 交叉验证-模型
adv_cv_results = lgb.cv(
    params,
    adv_data,
    num_boost_round=10000,
    nfold=5,
    categorical_feature=categorical_columns,
    early_stopping_rounds=200,
    verbose_eval=True,
    seed=42)

print('交叉验证中最优的AUC为 {:.5f}，对应的标准差为{:.5f}.'.format(
    adv_cv_results['auc-mean'][-1], adv_cv_results['auc-stdv'][-1]))

print('模型最优的迭代次数为{}.'.format(len(adv_cv_results['auc-mean'])))

# 通过对抗验证，我们发现模型的AUC达到了0.99。说明本次比赛的训练集和测试集的样本分布存在较大的差异。
# 然后，我们使用训练好的模型，对所有的样本进行预测，得到各个样本属于测试集的概率。这个之后会用到。
# ==========模型拟合&预测=========
params['n_estimators'] = len(adv_cv_results['auc-mean'])

model_adv = lgb.LGBMClassifier(**params)
model_adv.fit(df_adv.drop('Is_Test', axis=1), df_adv.loc[:, 'Is_Test'])

preds_adv = model_adv.predict_proba(df_adv.drop('Is_Test', axis=1))[:, 1]

## ================交叉验证（Cross Validation）===============
# 现在我们知道了训练集和测试集的分布存在很大的差异。那么接下来，我们采用交叉验证的方法，来评估模型的效果。
# 使用原始的df_train数据进行训练和验证集划分的交叉验证
def run_cv(df_train, sample_weight=None):
    if sample_weight is not None:
        train_set = lgb.Dataset(
            df_train.drop('HasDetections', axis=1),
            label=df_train.loc[:, 'HasDetections'], weight=sample_weight)

    else:
        train_set = lgb.Dataset(
            df_train.drop('HasDetections', axis=1),
            label=df_train.loc[:, 'HasDetections'])

    # Perform cross validation with early stopping
    params.pop('n_estimators', None)
    
    N_FOLDS = 5
    cv_results = lgb.cv(
        params,
        train_set,
        num_boost_round=10000,
        nfold=N_FOLDS,
        categorical_feature=categorical_columns,
        early_stopping_rounds=200,
        verbose_eval=True,
        seed=42)

    print('交叉验证中最优的AUC为 {:.5f}，对应的标准差为{:.5f}.'.format(
        cv_results['auc-mean'][-1], cv_results['auc-stdv'][-1]))

    print('模型最优的迭代次数为{}.'.format(len(cv_results['auc-mean'])))

    params['n_estimators'] = len(cv_results['auc-mean'])

    model_cv = lgb.LGBMClassifier(**params)
    model_cv.fit(df_train.drop('HasDetections', axis=1),
                 df_train.loc[:, 'HasDetections'])

    # AUC
    preds_test_cv = model_cv.predict_proba(
        df_test.drop('HasDetections', axis=1))[:, 1]
    auc_test_cv = roc_auc_score(df_test.loc[:, 'HasDetections'], preds_test_cv)
    print('模型在测试集上的效果是{:.5f}。'.format(
        auc_test_cv))

    return model_cv
# 调用模型
model_cv = run_cv(df_train)
## ==================在变量分布变化的情况下，除了交叉验证，还有哪些更优的方法？====================
### ================1.人工划分验证集===================
def run_lgb(df_train, df_validation):
    dtrain = lgb.Dataset(
        data=df_train.drop('HasDetections', axis=1),
        label=df_train.loc[:, 'HasDetections'],
        free_raw_data=False,
        silent=True)

    dvalid = lgb.Dataset(
        data=df_validation.drop('HasDetections', axis=1),
        label=df_validation.loc[:, 'HasDetections'],
        free_raw_data=False,
        silent=True)

    params.pop('n_estimators', None)

    clf = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=10000,
        valid_sets=[dtrain, dvalid],
        early_stopping_rounds=200,
        verbose_eval=True,
        categorical_feature=categorical_columns)

    params['n_estimators'] = clf.num_trees()

    model = lgb.LGBMClassifier(**params)
    model.fit(
        df_train.drop('HasDetections', axis=1),
        df_train.loc[:, 'HasDetections'])

    # AUC
    preds_test = model.predict_proba(
        df_test.drop('HasDetections', axis=1))[:, 1]
    auc_test = roc_auc_score(df_test.loc[:, 'HasDetections'], preds_test)
    print('模型在测试集上的效果是{:.5f}。'.format(
        auc_test))
    return model

# 之前已经用Date进行了排序，所以提取出后20%的样本作为验证集。
df_validation_1 = df_train.iloc[int(0.8 * len(df_train)):, ]
df_train_1 = df_train.iloc[:int(0.8 * len(df_train)), ]
# 方式1模型训练、验证与测试
model_1 = run_lgb(df_train_1, df_validation_1)


### ================2.使用和测试集最相似的样本作为验证集===================
# 提取出训练集上，样本是测试集的概率
df_train_copy = df_train.copy()
df_train_copy['is_test_prob'] = preds_adv[:len(df_train)]

# 根据概率排序-默认升序排列
df_train_copy = df_train_copy.sort_values('is_test_prob').reset_index(drop=True)

# 将概率最大的20%作为验证集
df_validation_2 = df_train_copy.iloc[int(0.8 * len(df_train)):, ]
df_train_2 = df_train_copy.iloc[:int(0.8 * len(df_train)), ]

df_validation_2.drop('is_test_prob', axis=1, inplace=True)
df_train_2.drop('is_test_prob', axis=1, inplace=True)

# 方式2模型训练、验证与测试
model_2 = run_lgb(df_train_2, df_validation_2)

### ================3.有权重的交叉验证 ===================
# 方式3模型训练、验证与测试
model_cv_wight = run_cv(df_train, sample_weight=preds_adv[:len(df_train)])


