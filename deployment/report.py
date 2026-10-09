"""Create a concise deployment report from completed local INT8 validations."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / 'deployment/history'


def main():
    selected = json.loads((HISTORY/'validated-matrix.json').read_text())
    rows = []
    for item in sorted(selected, key=lambda x:x['batch']):
        batch = item['batch']
        benchmark = json.loads((HISTORY/item['benchmark']).read_text())
        lifecycle = json.loads(Path(item['lifecycle']).read_text())
        assert benchmark['execution_pass'] and lifecycle['passed']
        power = benchmark['sustained_power']
        rows.append(dict(item,requests_per_second=power['requests_per_second'],
                         board_power_mean_w=power['power_w']['mean'],
                         board_memory_mib=power['board_memory_mib']['max'],
                         lifecycle_requests=len(lifecycle['first']),
                         partial_batch_tested='partial_batch' in lifecycle,
                         numerical_audits=[f'audit-ar-b{batch}.json',f'audit-acoustics-b{batch}.json']))
    complete = {row['batch'] for row in rows} == {1,2,4,8,16}
    report = dict(status='validated_complete' if complete else 'partial',precision='int8_smoothquant',
                  physical_gpu=1,sm=89,rows=rows,
                  timing='admission to last first-PCM in each wave; 5 warmups/30 waves',
                  throughput='30-second continuous first-chunk waves with cancellation',
                  limits=['First-chunk TensorRT coverage; out-of-profile tails use same-recipe Torch fallback',
                          'Numerical audit reports differences without a fixed tolerance gate; no ASR/MOS certification',
                          'CFM uses four estimator enqueues per call; full-solver compiler failure is archived'])
    (HISTORY/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    text = ['# IndexTTS INT8 SM89 本机结果','',
            '状态：'+('B1/2/4/8/16 全部通过本机验收。' if complete else '部分完成，其余 batch 仍在部署。'),'',
            '| Batch | 后端 | 首包 p50 ms | p95 ms | 首包 requests/s |',
            '|---|---|---:|---:|---:|']
    for row in rows:
        backend='native' if row['route'].startswith('native') else 'framework'
        text.append(f"| {row['batch']} | {backend} + Graph | {row['p50_ms']:.2f} | {row['p95_ms']:.2f} | {row['requests_per_second']:.2f} |")
    text.extend(['','延迟从入组到接收该 wave 最后一个首 PCM；5次预热、30波。吞吐为另行连续30秒首包测试，包含取消，不能当作完整语音吞吐。',
                 '', '完整EOS、运行中取消、逐字节重放与状态清理见 `history/lifecycle-selected-b*.json`；未满batch检查覆盖B>1。',
                 '权重身份与同配方数值审计见 `history/audit-*-b*.json`；误差仅报告，未作ASR/MOS音质认证。',
                 '首包使用本机TRT INT8；超出静态shape的尾包允许同配方Torch回退。CFM四步全图编译失败，使用既有四次estimator调用和外层Graph。',
                 '', '构建、验收与调用方式见 [部署说明](README.md)。'])
    (ROOT/'deployment/RESULTS.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(status=report['status'],batches=[row['batch'] for row in rows])))


if __name__=='__main__':
    main()
