"""Report only explicitly completed targets; never infer success from binaries."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
H=ROOT/'deployment/multibatch/history'


def main():
    rows=[]
    for b in (1,2,4,8,16,64,128):
        h=H/f'b{b}';migration=h/'migration-complete.json';complete=h/'optimization-complete.json'
        failure=h/'incomplete.json'
        marker=complete if complete.exists() else migration
        if marker.exists() and failure.exists() and failure.stat().st_mtime<marker.stat().st_mtime:
            archived=h/('resolved-incomplete-'+str(failure.stat().st_mtime_ns)+'.json');failure.rename(archived)
        if complete.exists():
            r=json.loads(complete.read_text());status='优化验收完成' if (h/'retained-execution-review.json').exists() else '初轮优化验收完成，剩余方向检查中';rows.append(r)
            selected=json.loads((h/'current-best-selected.json').read_text());baseline=json.loads((h/'migration-selected.json').read_text())
            r['changed_from_migration']=any(selected.get(k)!=baseline.get(k) for k in (set(selected)|set(baseline))-{'status'})
        elif migration.exists():r=json.loads(migration.read_text());status='迁移验收完成，待优化'
        else:r=dict(batch=b,status='incomplete');status='进行中/未验收'
        board=h/'WHITEBOARD.md'
        if board.exists():
            s=board.read_text();marker='\n## 当前结果\n'
            s=s.split(marker)[0]
            detail=''
            if complete.exists():
                a,z=r['original'],r['best'];detail=f"原/基线→最佳首包p50 {a['p50_ms']:.2f}→{z['p50_ms']:.2f}ms，p95 {z['p95_ms']:.2f}ms；平均/峰值功耗{z['mean_power_w']:.2f}/{z['peak_power_w']:.2f}W，首包{z['first_chunk_requests_s']:.2f}req/s。\n"
                if (h/'retained-execution-review.json').exists():
                    v=json.loads((h/'retained-execution-review.json').read_text());detail+='方向判据：'+', '.join(k+'='+d['status'] for k,d in v['target_decisions'].items())+'。实际覆盖/停止理由见retained-execution-review.json。\n'
            board.write_text(s+marker+'\n'+status+'。详见 '+('optimization-complete.json' if complete.exists() else 'migration-complete.json' if migration.exists() else '../active.json')+'。\n'+detail)
    residual_complete=(H/'fir-quant-matrix-decisions.json').exists()
    result=dict(status='validated_complete' if len(rows)==7 and residual_complete else 'initial_pass_validated_residual_review_pending' if len(rows)==7 else 'incomplete',completed_batches=[r['batch'] for r in rows],results=rows)
    (H/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    text=['# INT8 多batch结果','','下表为各档已验收的当前最佳；任务最终完成还需剩余FIR→INT8读写融合检查。' if not residual_complete else '七档迁移、优化及剩余FIR→INT8融合检查完成。','','| Batch | 原p50 ms | 最佳p50 ms | 最佳p95 ms | 平均W | 峰值W | 首包请求/s |','|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        a,b=r['original'],r['best'];text.append(f"| {r['batch']} | {a['p50_ms']:.2f} | {b['p50_ms']:.2f} | {b['p95_ms']:.2f} | {b['mean_power_w']:.2f} | {b['peak_power_w']:.2f} | {b['first_chunk_requests_s']:.2f} |")
    if residual_complete:
        text+=['','迁移后的新增保留：B1/B2/B4/B16使用burst1；B2/B128保留本档卷积tile方案；B4保留分组条件与latent prefix组合；B4/B64/B128保留72条FIR→原INT8量化融合，原FP32数学和尺度不变。B8没有保留额外候选，继续使用已验证的迁移方案。所有候选的接受/拒绝与实际层覆盖见各档retained-execution-review.json。','', '功耗为GPU1整卡NVML窗口平均/采样峰值；性能是整套方案收益，不能将其全部归因某一个kernel或batch变化。不同档位清单有区别，收益只用同档匹配对照。']
    unchanged=[str(r['batch']) for r in rows if not r['changed_from_migration']]
    if unchanged:text+=['','B'+ '/B'.join(unchanged)+' 当前没有保留迁移后的新方案；与迁移基线相同配置的测量差异属于波动，不计作新增优化收益。']
    text+=['','B1–B16原值是本次相邻重测旧引擎；B64/B128原值是各档迁移后的基线。五预热/30满batch首包；功耗/吞吐另测30秒，包含首包后的取消，不是完整语音吞吐。实际DSpark桥接+TRT+AOT/Triton，固定原INT8配方与保护浮点算子，形状外尾包同配方Torch回退。数值报告无固定L2阈值，无ASR/MOS认证。','','[运行与重建](README.md) · [详细记录](history/results.json)']
    (ROOT/'deployment/multibatch/RESULTS.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(status=result['status'],completed_batches=result['completed_batches'])))


if __name__=='__main__':main()
