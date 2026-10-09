"""Summarize B32 migration, retained optimization, execution and evidence limits."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
H=ROOT/'deployment/b32/history'


def main():
    original=json.loads((H/'delivery-migration-control-b32.json').read_text())
    best=json.loads((H/'delivery-current-best-b32.json').read_text())
    accepted=json.loads((H/'current-best.json').read_text())
    selected=json.loads((H/'current-best-selected.json').read_text())
    comparison=json.loads((H/'compare-delivery-b32.json').read_text())
    lifecycle=json.loads(Path(accepted['lifecycle']).read_text())
    smoke_path=H/'cli-smoke.json';smoke=json.loads(smoke_path.read_text()) if smoke_path.exists() else None
    assert original['execution_pass'] and best['execution_pass'] and lifecycle['passed']
    for key in ('device_round_fallbacks','native_cfm_fallbacks','native_vocoder_fallbacks'):
        assert best['measured_counters'][key]==0
    assert all(best['measured_counters']['head_graph_hits'][key]==30 for key in ('cfm','vocoder'))
    def metrics(x):
        v=x['sustained_power'];s=x['summary']['wave_admission_to_last_pcm_ms']
        return dict(p50_ms=s['p50'],p95_ms=s['p95'],first_chunk_requests_per_second=v['requests_per_second'],
            mean_board_power_w=v['power_w']['mean'],peak_board_power_w=v['power_w']['max'],board_memory_mib=v['board_memory_mib']['max'])
    a,b=metrics(original),metrics(best)
    report=dict(status='validated_complete' if smoke and smoke['passed'] else 'validated_engines_cli_pending',batch=32,precision='int8_smoothquant',physical_gpu=1,sm=89,
        original=a,retained_best=b,latency_p50_gain_pct=100*(a['p50_ms']-b['p50_ms'])/a['p50_ms'],
        first_chunk_throughput_gain_pct=100*(b['first_chunk_requests_per_second']/a['first_chunk_requests_per_second']-1),
        paired_mean_gain_95pct_ci_ms=comparison['metrics']['group_admission_to_last_pcm_ms']['paired_mean_gain_95pct_bootstrap_ci_ms'],
        selected=selected,acceptance=accepted,cli_smoke=smoke,
        backend='DSpark framework bridge + TensorRT11.3; custom AOT/Triton FIR and signed INT8 convolution inside Vocoder',
        retained=['Original source grouped conditions and latent prefix reuse after B32 migration comparison',
                  '109 complete-halo tiled FIR/Snake/FIR activations',
                  '76 compact-input signed INT8 implicit convolutions; original weight QuantizeLinear/scales/protection unchanged',
                  'Six C192/F1664/K11 convolution schedules changed to64/64/64 after bit-identical local probe and matched E2E'],
        not_retained=['Early quantization-only representation:1.09% local gain without expected materialization reduction',
                      'AR burst1/4, packing/latent-vector, flat projection: no consistent end-to-end evidence; burst2 retained'],
        residuals=['CFM has13 fused BF16 gemm_mha_v2 attention regions and59 actual INT8 tactics per estimator; four required sequential intervals retained',
                   'Protected floating convolutions/norms and remaining integer/floating compute are mandated by unchanged recipe',
                   'Remaining small runtime options were tested; further gains need new mechanism evidence, rather than repeated tactic searches'],
        limits=['Timing admission to last first-PCM in full B32 wave,5 warmups/30 waves, separate30-second first-chunk throughput including cancellation',
                'Static first-chunk coverage; out-of-profile tails use original same-recipe Torch fallback',
                'Floating numerical differences reported without fixed L2 gate; no ASR/MOS or acceptance-distribution certification'])
    (H/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    text=['# IndexTTS INT8 B32 本机结果','',
        '状态：'+('迁移、优化和实际CLI验收完成。' if report['status']=='validated_complete' else '引擎验收完成，CLI收尾中。'),'',
        '| 阶段 | 首包p50 ms | p95 ms | 首包requests/s | 平均整卡W | 采样峰值W |',
        '|---|---:|---:|---:|---:|---:|']
    for label,row in [('迁移基线',a),('保留最佳',b)]:text.append(f"| {label} | {row['p50_ms']:.2f} | {row['p95_ms']:.2f} | {row['first_chunk_requests_per_second']:.2f} | {row['mean_board_power_w']:.2f} | {row['peak_board_power_w']:.2f} |")
    ci=report['paired_mean_gain_95pct_ci_ms']
    text.extend(['',f"相邻、同条件30波：首包p50改善{report['latency_p50_gain_pct']:.2f}%，首包吞吐提升{report['first_chunk_throughput_gain_pct']:.2f}%；配对平均延迟收益95%区间{ci[0]:.2f}–{ci[1]:.2f}ms。",'',
        '边界是入组到该wave最后一个首PCM；5次预热、30波。另测30秒连续首包吞吐，包含取消，不能当作完整语音吞吐。功耗为GPU1整卡采样，未减去空闲功耗。',
        '', '保留源权重、SQalpha1.0和原BF16/FP32保护策略。Vocoder使用109个tiled FIR激活、76个真实signed INT8 MMA卷积，六个大通道卷积采用64/64/64 tile；其余五组件保留迁移的TRT计算。实际后端是DSpark桥接+TRT/AOT/Triton，未使用完整TRTLLM Executor。',
        '', '完整EOS、取消后同种子逐字节重放、清理及半满batch通过；AR/声学同配方审计与权重身份已检查。冻结PCM与原迁移结果存在浮点/编译策略差异，详见 [声学审计](history/'+Path(accepted['acoustic_audit']).name+')；数值误差仅报告，未做ASR/MOS认证。',
        '', 'CFM确认13个BF16注意力融合区域和59个INT8 tactic。四个季度区间保持，尾包超出静态形状时用同配方Torch回退。AR burst、打包和投影未出现一致收益，保留原配置。',
        '', '原始性能见 [基线](history/delivery-migration-control-b32.json)、[最佳](history/delivery-current-best-b32.json)、[配对比较](history/compare-delivery-b32.json)。[构建与运行](README.md) · [机制清单](history/mechanism-inventory.json) · [详细结果](history/results.json)。'])
    (ROOT/'deployment/b32/RESULTS.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(status=report['status'],original=a,retained_best=b),ensure_ascii=False))


if __name__=='__main__':main()
