# 输氢管道投产混气规律模拟

本项目用于第五届全国大学生油气储运工程数值仿真技能创新大赛“输氢管道投产混气规律模拟”赛题。程序以 12 km 输氢管道为对象，模拟入口通入 H2 推动初始 N2 隔离段和空气向出口置换的过程，输出 H2/N2/Air 摩尔分数分布、混气段长度、可燃风险段长度、有效 N2 隔离段长度、Froude 分层风险和推荐置换速度。

## 运行方式

安装依赖：

```bash
pip install -r requirements.txt
```

运行网页：

```bash
streamlit run streamlit_app.py
```

运行验证：

```bash
python run_validation.py
pytest -q
```

运行任务：

```bash
python run_task1_sweep.py
python run_task2_interrupt.py
```

生成局部 CFD/复核输入与结果：

```bash
python tools/build_external_cfd_case.py
python tools/run_local_buoyant_cfd2d.py --all
```

## 模型分层

项目分为四层，避免把展示图误当成 CFD：

1. **一维全管主模型**：负责 12 km 管道快速扫描，计算混气段、可燃风险段、有效 N2 隔离段和 Task1/Task2 指标。
2. **局部轴向放大**：从一维结果中截取界面附近曲线和色带，用于解释 H2/N2/Air 轴向界面，不是 CFD。
3. **局部二维浮升 CFD 复核**：`src/h2purge/local_cfd2d.py` 求解局部 x-z 窗口的低马赫浮升多组分输运，包含动量、浮力和 H2/N2/Air 组分输运，用于补充停输后截面分层风险判断。
4. **3D 管道动画**：网页里的三维管道是把一维摩尔分数映射成可视化动画，不是 CFD 求解结果。

当前局部 CFD 是二维 x-z 截面复核，不是完整三维可压缩多组分 CFD。完整三维 CFD 可进一步用 Fluent/OpenFOAM 建立圆管三维网格后，把结果放入 `outputs/cfd3d/<case>/` 供网页读取。

## 一维控制方程

主变量为摩尔分数/体积分数，不使用质量分数作为输出和安全判据变量：

```text
∂(C x_i)/∂t + ∂(u C x_i)/∂x = ∂/∂x (C K ∂x_i/∂x)
i = H2, N2, Air
C = p / (R T)
x_H2 + x_N2 + x_Air = 1
```

弥散系数采用工程等效湍流轴向弥散：

```text
K = D_mol + beta |u| D
```

默认 `beta = 0.5`，验证中给出 `beta = 0.2/0.5/0.8` 敏感性。

## 安全判据

氧气和惰性组分按空气折算：

```text
x_O2 = 0.21 x_Air
x_inert = x_N2 + 0.79 x_Air
```

保守可燃风险判据：

```text
0.04 <= x_H2 <= 0.75 且 x_O2 >= 0.05
```

停输阶段必须同时报告 Froude 分层风险。水平管停输时 `Fr = 0`，一维轴向模型不能替代截面分层分析。

## Task1

扫描参数：

```text
D = 0.7, 1.0, 1.2, 1.4 m
u = 5, 7, 9 m/s
p_back = 0.02, 0.05, 0.10 MPa
```

输出：

```text
outputs/task1/task1_summary.csv
outputs/task1/recommendation_table.csv
outputs/task1/*.png
```

推荐速度采用硬约束：

```text
dp / p_back < 0.10
```

在满足压降约束的速度中选 Fr 更大的速度。当前结果为 DN700 推荐 7 m/s，DN1000/DN1200/DN1400 推荐 9 m/s。

## Task2

固定工况：

```text
DN1200, u = 7 m/s, p_back = 0.10 MPa
```

在 H2 前缘到达 30%、60%、80% 管长时停输 5 min，输出停输前后混气段、有效 N2 隔离段、可燃风险段、Fr 和局部复核结果。

结论口径：一维模型中 5 min 轴向分子扩散导致的混气段增长很小；但停输时 `Fr = 0`，真实水平管内 H2 上浮富集和 Air/N2 下沉分层风险增强，因此需要局部二维/三维 CFD 或现场检测补充。

## 局部 CFD/复核结果

局部二维浮升 CFD 结果位于：

```text
outputs/cfd3d/DN1200_u7_p010_stop30_buoyant2d/
outputs/cfd3d/DN1200_u7_p010_stop60_buoyant2d/
outputs/cfd3d/DN1200_u7_p010_stop80_buoyant2d/
```

每个目录包含：

```text
metrics.json
xz_slice_h2.png
cross_section_h2.png
velocity_magnitude.png
flammable_region.png
cfd_result.vtk
```

网页只读取这些结果，不在前端临时伪造 CFD 云图。
