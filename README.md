# 输氢管道投产混气规律模拟

本项目用于“输氢管道投产混气规律模拟”赛题。程序以 12 km 输氢管道为对象，模拟入口通入 H2 推动初始 N2 隔离段和空气向出口置换的过程，输出 H2/N2/Air 摩尔分数分布、混气段长度、可燃风险段长度、有效 N2 隔离段长度、Froude 分层风险和推荐置换速度。

## 运行方式

安装依赖：

```bash
pip install -r requirements.txt
```

运行网页：

```bash
streamlit run streamlit_app.py
```

运行验证和任务：

```bash
python run_validation.py
pytest -q
python run_task1_sweep.py
python run_task2_interrupt.py
```

生成 CFD/复核相关文件：

```bash
python tools/build_external_cfd_case.py
python tools/run_local_buoyant_cfd2d.py --all
python tools/build_openfoam_3d_multispecies_case.py --all
```

## 架构原则

项目按层分工，避免把展示图误当成 CFD：

1. **一维全管主模型**：负责 12 km 全管置换、参数扫描、混气段、可燃风险段和有效 N2 隔离段。
2. **局部轴向放大**：从一维结果中截取界面附近曲线和色带，用于解释 H2/N2/Air 轴向界面，不是 CFD。
3. **局部二维浮升 CFD 复核**：`src/h2purge/local_cfd2d.py` 求解局部 x-z 窗口的低马赫浮升多组分输运，包含动量、浮力和 H2/N2/Air 组分输运。
4. **离线三维 CFD 工程包**：`tools/build_openfoam_3d_multispecies_case.py` 把一维停输场映射到圆管三维采样场，并完成 H2/N2/Air 到 H2/N2/O2 及质量分数的转换，供 Fluent/OpenFOAM 离线求解使用。
5. **网页展示层**：Streamlit 只做一维快速计算、动画展示和离线结果读取，不在前端临时伪造三维 CFD。

完整三维可压缩 CFD 应作为离线 Fluent/OpenFOAM 工程运行。跑完后把 `metrics.json`、云图和 VTK/VTU 放入 `outputs/cfd3d/<case>/`，网页自动读取展示。

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

## 安全判据

```text
x_O2 = 0.21 x_Air
x_inert = x_N2 + 0.79 x_Air
```

保守可燃风险判据：

```text
0.04 <= x_H2 <= 0.75 且 x_O2 >= 0.05
```

停输时 `Fr = 0`，一维轴向模型不能替代截面分层分析。

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

## 局部二维 CFD 复核结果

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

## 离线三维 CFD 工程包

```text
openfoam_cases/DN1200_u7_p010_stop30_rhoReacting3D/
openfoam_cases/DN1200_u7_p010_stop60_rhoReacting3D/
openfoam_cases/DN1200_u7_p010_stop80_rhoReacting3D/
```

每个目录包含：

```text
initial_3d_samples.csv
initial_3d_points.vtk
initial_3d_preview.png
metrics.json
openfoam_solver_notes.md
README.md
```

这些是离线三维 CFD 初始场和工程包，不是已经完成的 CFD 求解结果。外部 solver 真正跑完后，再把结果导入 `outputs/cfd3d/<case>/`。
