import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime
from gurobipy import Model, GRB

# 读取数据
file_path = "C:\\Users\\Jesen Jiang\\Desktop\\新1106汇报\\A-M32T monthly_sales.xlsx"
sales_data = pd.read_excel(file_path)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 将月份列转换为日期格式，并设置为索引
sales_data['Date'] = pd.to_datetime(sales_data['Date'], format='%Y-%m')
sales_data.set_index('Date', inplace=True)

# 获取销量列作为时间序列数据
time_series = sales_data['销量']

# 初始化预测列表和真实值列表
forecasts = []  # 用于存储测试集的预测值
forecast_values = []
actuals = []  # 用于存储测试集的真实值
fits_all_list = []  # 用于存储所有训练集的拟合值及其对应的日期

# 滚动预测，这里假设每1个月为一个周期（每预测新的一个月，训练集增加上一个月的真实值）
for i in range(24, len(time_series)):  # 从第25个月开始，每次增加1个月
    # 训练模型，使用前i个月的数据
    model = ExponentialSmoothing(time_series.iloc[:i], trend='mul', seasonal='mul', seasonal_periods=4)
    fit = model.fit()

    # 存储所有训练集的拟合值及其对应的日期
    fits_all_list.extend(zip(fit.fittedvalues.index, fit.fittedvalues.values))

    # 预测接下来1个月的数据
    forecast = fit.forecast(1)
    forecasts.append((time_series.index[i], forecast.iloc[0] if isinstance(forecast, pd.Series) else forecast[0]))
    forecast_values.append(forecast.iloc[0] if isinstance(forecast, pd.Series) else forecast[0])
    # 记录真实值，接下来的1个月
    actuals.append((time_series.index[i], time_series.iloc[i]))

# 将预测结果、真实值转换为DataFrame
forecast_df = pd.DataFrame(forecasts, columns=['Date', 'Forecast']).set_index('Date')
actual_df = pd.Series(dict(actuals)).rename('Actual')

# 将拟合值转换为DataFrame
fits_all_df = pd.DataFrame(fits_all_list, columns=['Date', 'Fitted']).set_index('Date')

# 确保fits_all与time_series的时间索引一致
fits_all_df = fits_all_df.loc[~fits_all_df.index.duplicated(keep='last')]
fits_all_df = fits_all_df.loc[fits_all_df.index < '2023-01-01']

# 筛选出21-22年的数据
start_date = '2021-01-01'
end_date = '2022-12-31'
fit_21_22 = fits_all_df[start_date:end_date]
actual_21_22 = time_series[start_date:end_date]

# 计算误差
error_21_22 = fit_21_22['Fitted'] - actual_21_22

# 创建21-22年的结果DataFrame
result_21_22 = pd.DataFrame({
    'Actual': actual_21_22,
    'Fitted': fit_21_22['Fitted'],
    'Error': error_21_22
})

# 计算23年的预测值
start_date_23 = '2023-01-01'
end_date_23 = '2023-12-31'
test_data_list = time_series[start_date_23:end_date_23]
forecast_23 = forecast_values
train_mean = actual_21_22.mean()
error_list = error_21_22.to_list()

def adjust_forecast(forecast_values, adjustment_factor):
    return [value * adjustment_factor for value in forecast_values]

def calculate_error(forecast_list, test_list):
    if len(forecast_list) != len(test_list):
        raise ValueError("两个输入列表长度必须相同")
        
    errors = []
    for forecast, test in zip(forecast_list, test_list):
        if test == 0:
            raise ValueError("测试数据列表中包含0值,不能进行除法操作")
        error = abs(forecast - test) / test
        errors.append(error)
    
    mean_error = sum(errors) / len(errors)
    return mean_error, errors

# 定义参数范围
adjustment_factors = np.arange(0.05, 1.05, 0.05)
thetas = np.arange(0.05, 1.05, 0.05)

# 存储结果
results = []
best_adjusted_forecasts = []
best_inventory = []

# 双层循环遍历参数
for adj_factor in adjustment_factors:
    for theta in thetas:
        # 应用调整因子
        adjusted_forecast_23 = adjust_forecast(forecast_23, adj_factor)
        
        # 计算调整后的预测误差
        mean_error_adj, errors_adj = calculate_error(adjusted_forecast_23, test_data_list)
        
        # 去除异常值（假设大于2的为异常值）
        errors_adj_filtered = [e for e in errors_adj if e <= 2]
        mean_error_adj_filtered = np.mean(errors_adj_filtered) if errors_adj_filtered else np.nan
        
        # 计算建议产量
        inventory = []
        for i in range(0, 12):
            right_demand = [adjusted_forecast_23[i] + error_list[j] for j in range(len(error_list))]
            delta = theta * adjusted_forecast_23[i]
            
            m = Model()
            x = m.addVar(lb=0, ub=1e6, vtype=GRB.INTEGER, name="x")
            y = {l: m.addVar(vtype=GRB.BINARY, name=f"y_{l}") for l in range(len(error_list))}
            
            m.setObjective(sum(y[l] for l in y) / len(y), GRB.MAXIMIZE)
            
            for l in range(len(error_list)):
                m.addConstr(x - right_demand[l] <= delta + (1 - y[l]) * 1e6, name=f"c1_{l}")
                m.addConstr(x - right_demand[l] >= -delta - (1 - y[l]) * 1e6, name=f"c2_{l}")
            
            m.setParam("OutputFlag", 0)
            m.optimize()
            
            optimal_x = int(round(m.getVarByName("x").X)) if m.status == GRB.OPTIMAL else np.nan
            inventory.append(optimal_x)
        
        # 计算建议产量的误差
        if all(not np.isnan(x) for x in inventory):
            mean_error_inv, errors_inv = calculate_error(inventory, test_data_list)
            errors_inv_filtered = [e for e in errors_inv if e <= 2]
            mean_error_inv_filtered = np.mean(errors_inv_filtered) if errors_inv_filtered else np.nan
        else:
            mean_error_inv = mean_error_inv_filtered = np.nan
        
        # 存储结果
        results.append({
            'adjustment_factor': adj_factor,
            'theta': theta,
            'mean_error_adj': mean_error_adj,
            'mean_error_adj_filtered': mean_error_adj_filtered,
            'mean_error_inv': mean_error_inv,
            'mean_error_inv_filtered': mean_error_inv_filtered,
            'adjusted_forecast': adjusted_forecast_23,
            'inventory': inventory
        })

# 转换为DataFrame并保存到Excel
results_df = pd.DataFrame(results)
results_df.to_excel('results.xlsx', index=False)

# 找出最佳参数组合（仅基于mean_error_inv_filtered最小）
valid_results = results_df[results_df['mean_error_inv_filtered'].notna()]
best_idx = valid_results['mean_error_inv_filtered'].idxmin()
best_row = valid_results.loc[best_idx]

# 输出最优参数和最小误差
print(f"最优参数: adjustment_factor={best_row['adjustment_factor']:.2f}, theta={best_row['theta']:.2f}")
print(f"最小mean_error_inv_filtered: {best_row['mean_error_inv_filtered']:.4f}")

# 使用最佳参数
best_adjusted_forecasts = best_row['adjusted_forecast']
best_inventory = best_row['inventory']

# 输出数组形式的调整后预测销量和建议产量
print("调整后预测销量数组:", best_adjusted_forecasts)
print("建议产量数组:", best_inventory)

# 使用最佳参数绘制图表
plt.figure(figsize=(14, 7))
plt.plot(forecast_df.index, test_data_list, label='真实销量', marker='o')
plt.plot(forecast_df.index, forecast_df['Forecast'], label='预测销量', marker='x', linestyle='--')
plt.plot(forecast_df.index, best_adjusted_forecasts, 
         label=f'调整后预测销量(adj={best_row["adjustment_factor"]:.2f})', 
         marker='x', linestyle="--", color='red')
plt.plot(forecast_df.index, best_inventory, 
         label=f'建议产量(theta={best_row["theta"]:.2f})', 
         marker='x', linestyle='--', color='green')
plt.title('A-M32T真实销量、预测销量、建议产量对比')
plt.xlabel('Year-Month')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()