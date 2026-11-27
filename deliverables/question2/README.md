# B 题第二问交付说明

本目录是第二问的导师审阅入口。交付内容覆盖十炉过程信号重建、氧枪流量计算、CO₂ 与 CO 有效时延估计、逐炉留一验证、模型比较、残差诊断、最终对齐数据和不确定性审计。现有 PDF、PNG 图件均保持生成时的版本。

## 1. 数据来源与字段

原始数据来自 B 题附件 1–10，对应炉号 6B03205、6B03206、6B03270、6B03271、6C02057、6C02058、6C02059、6C02060、6C02107 和 6C02108。每份工作簿包含采样序号、2047 个光谱强度列、累积耗氧量、CO₂ 和 CO。采样间隔按题面与连续序号统一为 0.5 s。

原始 XLSB 文件保存在本地 `data/raw/`，受 `.gitignore` 保护。仓库中的 `results/question2/data_diagnosis/data_quality_summary.csv` 记录文件名、样本数、字段名、缺失、负值、重复序号和突变候选。十炉主要过程字段没有缺失或重复序号。第 5 炉持续时间最短，且烟气突变候选较多，因此在解释窗口结果时单独关注。

## 2. 清洗与连续重建

读取程序兼容中文、英文和中英混合的序号及过程字段名称，并验证序号有限且连续。

- 累积耗氧量采用名义 10 s 节点的单调分段线性最小二乘拟合。相邻 0.5 s 区间的拟合增量除以时间间隔得到氧枪流量。
- CO₂ 和 CO 采用名义 5 s 节点的 Huber 稳健分段线性拟合，并施加非负约束。
- 首炉使用连续块留出检验 3 s 和 10 s 缺口下的重建误差，并检查峰值保留、非负性和累积耗氧量单调性。
- 程序保留真实记录中的零值，不逐点删除异常值。稳健损失用于降低大残差的权重。

十炉重建数据和基础过程图位于 `01_process_signals/`。首炉重建证据位于 `results/question2/heat_01_reconstruction/`。

## 3. 有效时延方法

正时延定义为分析仪观测时间晚于对应氧枪事件时间。本文估计的是包含反应、运输和测量链影响的有效时延，数据无法把这些组成部分分别识别。

局部时延在 120 s 窗口内估计，窗口步长为 30 s。每个窗口对氧枪流量和烟气信号做线性去趋势，再在 0–60 s 范围搜索互相关峰。可靠窗口须同时满足相关系数不低于 0.45、峰值相对该窗口相关中位数的对比度不低于 0.08、氧枪流量标准差不低于 0.10，并且峰值不位于搜索边界。

十炉共得到 67 个可靠 CO 窗口和 57 个可靠 CO₂ 窗口，两种气体均覆盖全部十炉。完整窗口、互相关曲线、筛选结果和逐炉数据位于 `02_delay_model/delay_estimation_evidence/`。

## 4. LOHO 验证与模型比较

验证采用逐炉留一法。每折使用九炉训练并在剩余一炉测试。固定时延、单调递减线性、反比例、单调回归和三档流量模型所需的参数、分组边界与组内代表值只使用训练炉估计。主指标为十个测试炉 RMSE 的等权平均，避免可靠窗口较多的炉次主导模型选择。

基础四模型比较中，CO 的固定、单调线性、反比例和单调回归平均炉 RMSE 分别为 15.02、15.80、15.51 和 15.65 s。CO₂ 对应结果为 16.49、18.35、17.63 和 17.76 s。扩展三模型比较中，CO 的固定、三档流量和单调线性结果为 15.02、15.15 和 16.73 s，CO₂ 为 16.49、16.24 和 18.35 s。

综合跨炉稳定性，最终固定基线采用 CO 32 s 和 CO₂ 34 s。流量相关候选作为受支持范围内的扩展实验，未替代固定基线。模型表、逐炉指标和测试预测位于 `03_uncertainty/loho_validation/`。

## 5. 残差与稳健性诊断

残差定义为预测时延减去窗口估计时延。`03_uncertainty/` 提供观测与预测对照、分模型残差分布、逐炉误差热图、流量分组审计、阈值敏感性、炉内与炉间关系、极端峰值和留一炉稳健性结果。

CO 的 pooled Spearman 相关为 -0.424，方向符合流量增大、时延缩短的机理预期。逐炉内部方向和强度并不一致。CO₂ 的 pooled Spearman 相关为 -0.034，没有显示稳定的单调关系。这些结果支持将固定时延作为跨炉基线，并把流量关系作为需要额外状态变量和独立验证的候选。

## 6. 最终对齐结果

最终数据位于 `04_alignment/final_aligned_dataset/`。CO 在动态候选有效且事件到分析仪的时间映射严格递增时使用受限动态值，否则整炉回退到 32 s。CO₂ 使用 34 s 固定基线。对齐不进行端点外推。

现有验证中，第 1、3、4、6 炉使用 CO 动态候选，第 2、5、7、8、9、10 炉使用固定回退。十炉 CO 与 CO₂ 累计量均保持非递减。定稿对齐图位于 `04_alignment/final_figures/`。

## 7. 局限性

- 窗口时延标签来自互相关峰，缺少独立运输时延或反应时延测量。
- 可靠窗口彼此重叠，窗口数量不能解释为完全独立的观测数量。
- 首炉参与流程设计，首炉留出实验用于重建方法检查，不构成完全独立的外部验证。
- 不同炉次的阶段、反应状态和未观测变量会影响局部相关峰，单一流量变量不足以解释全部动态变化。
- 累积量之间的高相关包含共同时间趋势，应结合对齐后的瞬时流量关系和跨炉总量检验解释。
- 气体流量单位按题面记录为 m³/s。提交论文时应在数据说明中明确题面出处与原表字段的对应关系。

## 8. 复现

在仓库根目录创建包含 `requirements-question2.txt` 所列依赖的 Python 环境，并把附件 1–10 放入 `data/raw/`。

```powershell
python -m pytest -q tests/test_delay_interface.py tests/test_ten_heat_loader.py

python src/question2/ten_heat_reconstruction_delay.py `
  --raw-dir data/raw `
  --output-dir results/question2/ten_heat_reconstruction_delay_rerun

python src/question2/delay_model_v2.py `
  --input results/question2/ten_heat_reconstruction_delay_rerun/all_window_delay_estimates.csv `
  --output-dir results/question2/delay_model_v2_rerun

python src/question2/final_process_relationships.py `
  --input-dir results/question2/ten_heat_reconstruction_delay_rerun `
  --output-dir results/question2/final_process_relationships_rerun

python src/question2/run_dynamic_delay_validation_corrected.py
```

各脚本拒绝覆盖已有输出。重复运行时使用新的输出目录。2026-11-28 的复现检查中，十炉重建的 55 个文件、过程关系分析的 39 个文件和修正动态验证的 16 个文件均与归档交付版本逐文件一致。时延接口与十炉字段读取测试共 9 项。

## 9. 目录索引

- `01_process_signals/`：十炉六信号数据和汇总图。
- `02_delay_model/`：时延接口、窗口估计证据与修正动态验证。
- `03_uncertainty/`：LOHO、残差、误差热图和稳健性审计。
- `04_alignment/final_aligned_dataset/`：十炉最终对齐数据和质量审计。
- `04_alignment/final_figures/`：十炉定稿对齐图。
- `05_documentation/file_manifest.csv`：正式交付文件的相对路径、大小和 SHA-256。
