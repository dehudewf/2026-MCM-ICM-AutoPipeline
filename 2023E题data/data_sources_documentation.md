
# 真实地点数据来源说明

## 数据来源概述

本数据基于四个真实地点的特征生成，参考了多个权威数据源的典型值范围。

## 四个地点说明

### 1. 美国大峡谷国家公园（Grand Canyon National Park）
- **类型**: 受保护区域
- **特征**: 国际暗空公园，Bortle等级1-3级
- **数据来源**: 
  - Grand Canyon International Dark Sky Park Nomination Package
  - VIIRS Nighttime Light Data (NASA/NOAA)

### 2. 中国西部小镇
- **类型**: 农村社区
- **特征**: 人口密度较低，光污染程度较低
- **数据来源**:
  - 中国统计年鉴
  - 省级统计年鉴
  - 夜间灯光数据作为人口密度代理

### 3. 伦敦周边住宅区（London Suburban）
- **类型**: 郊区社区
- **特征**: Bortle 5-7级，中等光污染水平
- **数据来源**:
  - Light Pollution Map - 全球夜空亮度可视化
  - ONS Crime Statistics (英国)
  - VIIRS 夜间灯光数据

### 4. 美国纽约曼哈顿（Manhattan, NYC）
- **类型**: 城市社区
- **特征**: 高密度城市，光污染极强
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
- **S1 人口密度**: U.S. Census Bureau / 中国统计年鉴
- **S2 社会发展水平**: BEA / ACS / 世界银行
- **S3 交通事故率**: NHTSA - FARS / WHO Global Status Report on Road Safety
- **S4 犯罪率**: FBI Crime Data Explorer / ONS / 国家统计年鉴

### 生态影响指标
- **N1 植被覆盖率**: USGS - NLCD / MODIS NDVI / ESA Land Cover
- **N2 生物多样性**: GBIF / NatureServe / IUCN Red List
- **N3 气候(湿度/透明度)**: NOAA - NCEI / ECMWF
- **N4 地理(日照/海拔)**: USGS Elevation / NOAA

### 能源因素指标
- **E1 单位面积照明能耗**: U.S. EIA / VIIRS夜光数据
- **E2 照明系统效率**: DOE - SSL Program / 城市照明改造报告

## 数据获取注意事项

1. **VIIRS夜间灯光数据**: 需要从NASA/NOAA下载原始辐射值并进行GIS处理
2. **植被覆盖数据**: 需要从MODIS或ESA Land Cover数据集提取
3. **生物多样性数据**: 需要从GBIF或IUCN获取物种分布数据
4. **社会经济数据**: 需要从各国统计年鉴或国际组织数据库获取
5. **犯罪和交通事故数据**: 需要从各国官方统计部门获取

## 数据准确性说明

本数据基于各地点的典型特征和公开数据源的合理范围生成。
如需获取精确的原始数值，建议：
1. 从上述数据源下载原始数据
2. 使用GIS工具提取各地点对应的数据
3. 进行数据清洗和标准化处理
4. 验证数据的准确性和一致性
