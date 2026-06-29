# 输氢管道投产混气规律模拟

本项目用于第五届全国大学生油气储运工程数值仿真技能创新大赛“输氢管道投产混气规律模拟”赛题。对象为 12 km 输氢管道：初始入口侧 8% 管长为 N2 隔离段，其余为空气；t=0 后入口通入纯 H2，模拟 H2 推动 N2 隔离段完成全线置换的过程。

## 安装

```bash
python -m pip install -r requirements.txt
```

## 运行

```bash
python run_validation.py
python run_task1_sweep.py
python run_task2_interrupt.py
python tools/build_external_cfd_case.py
streamlit run streamlit_app.py
```

脚本输出写入：

- `outputs/validation`：解析解、守恒、网格无关性、数值弥散、beta 敏感性验证。
- `outputs/task1`：36 组管径/速度/背压扫描结果与推荐速度表。
- `outputs/task2`：30%、60%、80% 中断工况结果。
- `cfd_cases`：外部 CFD 初始化输入包，不是 CFD 求解结果。
- `outputs/cfd3d`：外部 Fluent/OpenFOAM 等求解器输出结果的导入目录。

## 模型口径

主变量为摩尔分数/体积分数：

```text
x_H2, x_N2, x_Air
```

不是质量分数。每个网格每个时刻满足：

```text
x_H2 + x_N2 + x_Air = 1
```

一维弱可压多组分对流-弥散方程写为：

```text
∂(C x_i)/∂t + ∂(u C x_i)/∂x = ∂/∂x (C K ∂x_i/∂x)
```

其中 `C = p/(RT)`，`x_i` 为组分摩尔分数。当前 V1 求解器在组成方程中推进 `x_i`，压力用于计算密度、Re、Fr 和摩阻压降诊断。程序内部压力单位统一为 Pa 绝压；脚本和网页中的 MPa 背压输入按绝压处理。

## 弥散模型

主模型不只使用分子扩散，而采用湍流轴向等效弥散：

```text
K = D_mol + beta |u| D
```

默认 `beta = 0.5`。停输阶段 `u=0` 时，Task2 使用分子扩散系数。`run_validation.py` 会输出 `outputs/validation/beta_sensitivity.csv`，对 `beta = 0.2/0.5/0.8` 做敏感性分析。

在 `K = beta u D` 口径下，相同置换进度的界面展宽主要由管径 D 和 beta 控制，速度项在 `K t` 中近似抵消。因此 Task1 的混气段规律应解释为管径主导；速度推荐来自 Fr 分层风险和摩阻压降约束的权衡，而不是“速度越大混得越少”。

## 安全判据

```text
x_O2 = 0.21 x_Air
x_inert = x_N2 + 0.79 x_Air
```

保守可燃风险判据：

```text
0.04 <= x_H2 <= 0.75 且 x_O2 >= 0.05
```

同时输出密度 Froude 数 Fr：

- `Fr < 1`：强分层风险。
- `1 <= Fr < 3`：中等分层风险，一维近似需要谨慎。
- `Fr >= 3`：分层风险较低。

水平管停输时 `Fr=0`，一维轴向模型不能替代三维截面分层风险分析。

## Task1

参数扫描：

- 管径：DN700、DN1000、DN1200、DN1400。
- 置换速度：5、7、9 m/s。
- 出口背压：0.02、0.05、0.10 MPa。

输出 `outputs/task1/task1_summary.csv` 和 `outputs/task1/recommendation_table.csv`。推荐速度采用硬约束：

```text
dp / p_back < 0.10
```

在满足压降约束的速度中选 Fr 最大者。当前结果为：

- DN700：推荐 7 m/s。
- DN1000、DN1200、DN1400：推荐 9 m/s。

## Task2

固定工况：

```text
DN1200, u = 7 m/s, p_back = 0.10 MPa
```

分别在 H2 前缘到达 30%、60%、80% 管长时停输 5 min。输出停输前后混气段、有效 N2 隔离段、可燃风险段、Fr 和重力流侵入量级估算。

结论口径必须保持清楚：一维模型中 5 min 轴向分子扩散导致的混气段增长很小；但停输时 Fr=0，真实水平管内 H2 上浮富集和 Air/N2 下沉分层风险增强，需要局部二维/三维 CFD 或现场检测补充。

## 外部 CFD

网页不内置 CFD 求解器，也不会把一维可视化图伪装成 CFD。流程是：

```text
1D 全管模型 -> cfd_cases 外部 CFD 初始场 -> Fluent/OpenFOAM 求解 -> outputs/cfd3d 导入展示
```

生成外部 CFD 输入包：

```bash
python tools/build_external_cfd_case.py
```

真实 CFD 求解完成后，把结果放入：

```text
outputs/cfd3d/DN1200_u7_p010_stop60/
  metrics.json
  xz_slice_h2.png
  cross_section_h2.png
  flammable_region.png
  cfd_result.vtu 或 cfd_result.vtk（可选）
```

没有这些外部结果时，网页只显示“未导入外部三维 CFD 结果”，不会画假云图。
