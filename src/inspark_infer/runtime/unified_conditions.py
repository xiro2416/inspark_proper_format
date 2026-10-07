"""Opt-in acoustic-condition batching using existing sample-local torch ops.

Only equal speech lengths are combined. GroupNorm reduces channels/time within
each sample, so padding a short utterance up to another utterance's length would
change its conditioning. This module deliberately never does that.
"""
from __future__ import annotations

import torch
from torch import nn


def _validate_modules(quantizer, projection, regulator):
    from inspark_infer.models.indextts2.upstream.s2mel.modules.length_regulator import InterpolateRegulator
    from inspark_infer.models.indextts2.upstream.utils.maskgct.models.codec.amphion_codec.quantize.residual_vq import ResidualVQ
    if (not isinstance(quantizer,ResidualVQ) or quantizer.num_quantizers!=1
            or quantizer.quantizer_type!='fvq'):
        raise ValueError('Batched conditions require the single factorized semantic codebook')
    if any(module.training for root in (quantizer,projection,regulator) for module in root.modules()):
        raise ValueError('Batched conditions require evaluation mode')
    if not all(isinstance(module,(nn.Sequential,nn.Linear)) for module in projection.modules()):
        raise ValueError('Unexpected GPT condition projection; sample independence is not established')
    if (not isinstance(regulator,InterpolateRegulator) or regulator.is_discrete
            or not regulator.interpolate or hasattr(regulator,'vq') or regulator.f0_condition):
        raise ValueError('Batched conditions require the continuous interpolation regulator without VQ/F0')


def grouped_conditions(tts,code_tensors,latents,speech_counts,*,phase=None,streams=(),graph_bank=None,flat_projection=False):
    """Return one [1,F,C] view per input request, in original order.

    Codec codes are [quantizer,B,T], not [B,quantizer,T]. The existing serial
    path uses [1,1,T], which cannot reveal this otherwise important distinction.
    Known host speech lengths provide interpolation sizes without device scalar
    reads. Batch shape changes can still change floating-point reduction order;
    the enabled candidate requires the usual GPU arithmetic/audio audit.
    """
    if not len(code_tensors)==len(latents)==len(speech_counts):
        raise ValueError('Condition input counts differ')
    if not code_tensors:return [],[]
    quantizer=tts.semantic_codec.quantizer
    projection=tts.s2mel.models['gpt_layer']
    regulator=tts.s2mel.models['length_regulator']
    _validate_modules(quantizer,projection,regulator)
    groups={}
    for index,(codes,latent,count) in enumerate(zip(code_tensors,latents,speech_counts)):
        if (type(count) is not int or count<1 or codes.ndim!=2 or tuple(codes.shape)!=(1,count)
                or latent.ndim!=3 or tuple(latent.shape[:2])!=(1,count) or codes.device!=latent.device):
            raise ValueError('Expected matching unpadded one-request code and latent sequences')
        key=(count,codes.dtype,latent.dtype,codes.device,latent.shape[-1])
        groups.setdefault(key,[]).append(index)
    projected=None
    if flat_projection:
        if graph_bank is not None:raise ValueError('Flat projection must not repeat projection in a condition Graph')
        def project():return projection(torch.cat([value[0] for value in latents],0))
        packed=project() if phase is None else phase('condition_projection_flat',len(latents),project)
        projected=[value.unsqueeze(0) for value in packed.split(speech_counts,dim=0)]
    outputs=[None]*len(code_tensors);inventory=[]
    parent=torch.cuda.current_stream() if streams else None
    for stream in streams:stream.wait_stream(parent)
    try:
        for group_index,((count,_,_,device,_),indices) in enumerate(groups.items()):
            frames=int(count*1.72)
            def compute():
                if graph_bank is not None:
                    result=graph_bank.run([code_tensors[index] for index in indices],
                                         [latents[index] for index in indices],count)
                    if result is not None:return result
                codes=torch.cat([code_tensors[index] for index in indices],0)
                condition_latent=(projection(torch.cat([latents[index] for index in indices],0)) if projected is None else
                                  torch.cat([projected[index] for index in indices],0))
                lookup=getattr(tts,'projected_vq_table',None)
                embedding=(quantizer.vq2emb(codes.unsqueeze(0)).transpose(1,2)
                           if lookup is None else lookup(codes))
                semantic=embedding+condition_latent
                lengths=torch.full((len(indices),),frames,device=device,dtype=torch.long)
                return regulator(semantic,ylens=lengths,n_quantizers=3,f0=None,length_hint=frames)[0]
            if streams:
                stream=streams[group_index%len(streams)]
                with torch.cuda.stream(stream):
                    for index in indices:
                        code_tensors[index].record_stream(stream)
                        latents[index].record_stream(stream)
                        if projected is not None:projected[index].record_stream(stream)
                    condition=(compute() if phase is None else
                               phase('condition_group',len(indices),compute,speech_codes=count,frames=frames))
                    condition.record_stream(parent)
            else:
                condition=(compute() if phase is None else
                           phase('condition_group',len(indices),compute,speech_codes=count,frames=frames))
            if tuple(condition.shape[:2])!=(len(indices),frames):
                raise RuntimeError('Condition regulator changed the expected exact-length extent')
            for row,index in enumerate(indices):outputs[index]=condition[row:row+1]
            inventory.append(dict(speech_codes=count,frames=frames,batch=len(indices),request_indices=list(indices)))
    finally:
        for stream in streams:parent.wait_stream(stream)
    return outputs,inventory
