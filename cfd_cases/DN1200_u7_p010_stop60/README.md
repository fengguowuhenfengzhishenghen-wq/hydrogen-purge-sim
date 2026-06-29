# 外部 CFD 输入数据包

本目录由已经验证的一维 H2/N2/Air 置换模型生成，用于给 Fluent、OpenFOAM 等外部 CFD 工程提供初始浓度场。
注意：这里不是 CFD 求解结果，不能把这些文件当成 CFD 云图导入 Streamlit 的“外部三维 CFD 复核结果”区域。

文件含义：

- `case_metadata.json`：工况、停输位置、前缘位置和局部窗口信息。
- `full_pipe_stop_profile.csv`：60% 中断瞬间 12 km 全管一维摩尔分数场。
- `windows/*/initial_1d_profile.csv`：局部 CFD 窗口的一维轴向初始浓度。
- `windows/*/initial_field_points.vtk`：把一维初始场映射到圆截面点云，方便 ParaView 检查或二次映射。
- `windows/*/initial_profile_preview.png`：局部窗口初始浓度预览图，不是 CFD 结果图。

建议的 CFD 复核目标：

验证一维模型无法解析的停输截面分层风险，包括 H2 上浮富集、Air/N2 下沉、
以及停输 300 s 后横截面内是否形成局部可燃区域。

只有当外部求解器真正输出 `metrics.json`、`xz_slice_h2.png`、`cross_section_h2.png`、
`flammable_region.png` 或 VTK/VTU 结果文件后，才能称为外部 CFD 复核结果。

工况：DN1200_u7_p010_stop60
中断位置：60% L
停输时刻：1028.571 s
H2 前缘：7202.313 m
N2/Air 前缘：8162.311 m