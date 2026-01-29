
# 美国境内四个地点数据来源说明

## 数据来源概述

本数据基于美国境内四个真实地点的特征生成，参考了多个权威数据源的典型值范围。

## 四个地点说明

### 1. 美国大峡谷国家公园（Grand Canyon National Park）
- **类型**: 受保护区域
- **州**: Arizona
- **特征**: 国际暗空公园，Bortle等级1-3级
- **数据来源**: 
  - Grand Canyon International Dark Sky Park Nomination Package
  - VIIRS Nighttime Light Data (NASA/NOAA)

### 2. 南达科他州麦库克县（McCook County, South Dakota）
- **类型**: 农村地区
- **州**: South Dakota
- **特征**: 美国中北部平原农业县，人口密度低，植被覆盖率高
- **数据来源**:
  - U.S. Census Bureau (人口密度)
  - USDA (农业和植被数据)
  - South Dakota Department of Transportation (交通事故率)
  - FBI Crime Data Explorer (犯罪率)

### 3. 纽约州斯卡斯代尔（Scarsdale, New York）
- **类型**: 郊区社区
- **州**: New York
- **特征**: 纽约州韦斯特切斯特县富裕郊区，人口密度中等
- **数据来源**:
  - U.S. Census Bureau (人口密度)
  - Westchester County Planning Department
  - New York State Department of Transportation
  - Westchester County District Attorney (犯罪率)

### 4. 美国纽约曼哈顿（Manhattan, NYC）
- **类型**: 城市社区
- **州**: New York
- **特征**: 纽约市中心，高密度城市（光污染极强）
- **数据来源**:
  - VIIRS Nighttime Light Data (NASA/NOAA)
  - FBI Crime Data Explorer
  - U.S. Census Bureau

## 指标数据来源

### 人类健康影响指标
- **H1 昼夜节律失调**: CDC - NHANES (国家健康与营养调查)
- **H2 代谢与肥胖指标**: CDC - NHANES
- **H3 视功能损害**: CDC - Vision Health Data
- **H4 心理与情绪**: SAMHSA / BRFSS

### 社会影响指标
- **S1 人口密度**: U.S. Census Bureau
- **S2 社会发展水平**: BEA / ACS
- **S3 交通事故率**: NHTSA - FARS / 各州交通部门
- **S4 犯罪率**: FBI Crime Data Explorer / 各州执法部门

### 生态影响指标
- **N1 植被覆盖率**: USGS - NLCD / USDA
- **N2 生物多样性**: USGS / NatureServe
- **N3 气候(湿度/透明度)**: NOAA - NCEI
- **N4 地理(日照/海拔)**: USGS Elevation / NOAA

### 能源因素指标
- **E1 单位面积照明能耗**: U.S. EIA / VIIRS夜光数据
- **E2 照明系统效率**: DOE - SSL Program / 各州能源部门

## 数据准确性说明

本数据基于各地点的典型特征和公开数据源的合理范围生成。
如需获取精确的原始数值，建议：
1. 从上述数据源下载原始数据
2. 使用GIS工具提取各地点对应的数据
3. 进行数据清洗和标准化处理
4. 验证数据的准确性和一致性
