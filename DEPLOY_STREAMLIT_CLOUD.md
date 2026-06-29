# Streamlit 公网部署说明

本地地址 `http://localhost:8505` 只能在当前电脑打开。手机、平板或评委电脑要访问，需要把项目部署到公网平台，推荐使用 Streamlit Community Cloud。

## 1. 推荐部署方式

部署完成后会得到类似下面的公网链接：

```text
https://your-app-name.streamlit.app
```

这个链接可以直接发给手机、平板或评委电脑打开。

## 2. 上传到 GitHub

建议把 `hydrogen_purge_sim` 作为一个独立仓库上传。仓库根目录至少包含：

```text
streamlit_app.py
requirements.txt
packages.txt
.streamlit/config.toml
src/
outputs/
cfd_cases/
openfoam_cases/
README.md
```

如果上传的是上一层 `ppt` 文件夹，则 Streamlit Cloud 的入口文件填写：

```text
hydrogen_purge_sim/streamlit_app.py
```

如果上传的是 `hydrogen_purge_sim` 本身，则入口文件填写：

```text
streamlit_app.py
```

## 3. Streamlit Cloud 设置

1. 打开 `https://share.streamlit.io/`。
2. 使用 GitHub 登录。
3. 点击 `Create app`。
4. 选择仓库和分支。
5. 填写入口文件路径。
6. Python 版本建议选择 3.11 或 3.12。
7. 部署完成后复制生成的 `https://xxx.streamlit.app` 链接。

## 4. 依赖文件

`requirements.txt` 安装 Python 依赖：

```text
numpy
scipy
pandas
matplotlib
streamlit
pytest
```

`packages.txt` 安装中文字体：

```text
fonts-noto-cjk
```

该字体用于保证云端图表里的中文标题、图例、坐标轴正常显示。

## 5. 当前网页能展示的内容

- 一维轴向对流-弥散主模型。
- 动态置换模拟与局部轴向放大。
- 任务 2 中断工况说明。
- 外部 CFD 初始场输入包。
- `outputs/cfd3d` 中的局部分层复核图片。
- OpenFOAM `scalarTransportFoam` 导出的 300 s VTU 体数据。

注意：网页中的 3D 管道动画是一维摩尔分数场的可视化映射，不是三维 CFD 求解。OpenFOAM 标量输运结果用于复核局部输运，不等同于完整的可压缩多组分重力 CFD。

## 6. 提交前检查

部署前建议在本地运行：

```bash
python -m py_compile streamlit_app.py src/h2purge/cfd_import.py
python -m pytest -q
streamlit run streamlit_app.py
```

打开网页后重点检查：

- 默认参数为 DN1200、7 m/s、0.10 MPa、dx=10 m。
- 侧边栏外部复核 case 默认选中 `DN1200_u7_p010_stop60_reduced2d`。
- 外部复核区显示 OpenFOAM 检查状态、VTU 路径和下载按钮。
- 页面没有把一维可视化误称为完整三维 CFD。
