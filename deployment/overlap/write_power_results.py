"""Attach measured sustained-window power without altering earlier latency evidence."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent;history=root/'history'
a=json.loads((history/'b16-power.json').read_text());b=json.loads((history/'b32-power.json').read_text())
assert a['status']==b['status']=='probe_validated'
rows={label:report['sustained'][mode] for label,report,mode in [('B16串行两批（第二bank驻留）',a,'resident_one_bank_serial'),('双B16并行',a,'two_bank_concurrent'),('单B32',b,'single_bank')]}
for row in rows.values():assert row['seconds']>=30 and row['sensor']['sensor_samples']>=100
result=dict(status='validated',physical_gpu=1,scope='Three separately measured 30-second back-to-back first-chunk serving windows. Loading/warmup/hash excluded; dispatch and cancellation included; board power includes pre-existing idle allocation. Not power from original latency runs.',modes=rows)
(history/'power-results.json').write_text(json.dumps(result,indent=2)+'\n')
text=['## 三种调度的功耗补测','', 'GPU1，预热后各连续运行至少30秒。20ms间隔采样整卡瞬时功耗，包含请求派发/取消，排除加载、预热和结果哈希；本次功耗是补测，不与原时延记录假称同一窗口。', '', '| 调度 | 平均W | 峰值W | 样本数 | 测量秒数 |', '|---|---:|---:|---:|---:|']
for label,row in rows.items():
 s=row['sensor'];text.append(f"| {label} | {s['power_w_mean']:.2f} | {s['power_w_peak']:.2f} | {s['sensor_samples']} | {row['seconds']:.2f} |")
text.extend(['', '每轮完整交付32请求首包，随后取消；原生CFM/Vocoder Graph计数完整，无AR/CFM/Vocoder回退，取消后无残留sessions/active_rows/errors。B16测量保持双模型驻留，B32为单模型。不同batch生成行为差异沿用上文边界。', '', '[原始B16功耗](history/b16-power.json) · [原始B32功耗](history/b32-power.json) · [功耗汇总](history/power-results.json)'])
p=root/'RESULTS.md';s=p.read_text();marker='## 三种调度的功耗补测';s=s.split(marker)[0].rstrip()+'\n\n'+'\n'.join(text)+'\n';p.write_text(s)
p=root/'WHITEBOARD.md';s=p.read_text();s+='\n功耗补测完成：同调度GPU1预热后连续30秒、20ms整卡瞬时采样；加载/预热/哈希排除，真实派发和取消计入。'+ '；'.join(f"{label}平均{row['sensor']['power_w_mean']:.2f}W/峰值{row['sensor']['power_w_peak']:.2f}W" for label,row in rows.items())+'。原生工作完整、无回退或残留请求；见history/power-results.json，非原始时延窗口。\n';p.write_text(s)
for label,row in rows.items():print(label,row['sensor'])
