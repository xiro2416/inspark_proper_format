"""Original ZipVoice audio/text conditioning, with explicit engine-profile checks."""
import json
from pathlib import Path

def prepare(bundle,manifest,prompt_wav,prompt_text,text,output):
 import torch,torchaudio,soundfile as sf
 from lhotse.utils import compute_num_frames
 from safetensors.torch import save_file
 from inspark_infer.models.zipvoice.tokenizer import EmiliaTokenizer
 from inspark_infer.build.zipvoice import check_workload
 wave,sr=sf.read(prompt_wav,dtype='float32',always_2d=True)
 waveform=torch.from_numpy(wave.mean(axis=1)).unsqueeze(0)
 if sr!=24000:waveform=torchaudio.functional.resample(waveform,sr,24000)
 rms=waveform.square().mean().sqrt()
 if not torch.isfinite(rms) or rms<=0:raise ValueError('Reference audio must be finite and nonzero')
 if rms<.1:waveform=waveform*(.1/rms)
 transform=torchaudio.transforms.MelSpectrogram(sample_rate=24000,n_fft=1024,hop_length=256,n_mels=100,center=True,power=1)
 mel=transform(waveform).clamp(min=1e-7).log().squeeze(0).T.contiguous()
 frames=compute_num_frames(waveform.shape[1]/24000,256/24000,24000)
 if mel.shape[0]>frames:mel=mel[:frames]
 elif mel.shape[0]<frames:mel=torch.nn.functional.pad(mel.T.unsqueeze(0),(0,frames-mel.shape[0]),mode='replicate').squeeze(0).T.contiguous()
 if frames!=375:raise ValueError(f'Reference produces {frames} frames; this release requires 375. No implicit VAD/cropping.')
 tokenizer=EmiliaTokenizer(str(Path(bundle)/'config/tokens.txt'))
 prompt,target=tokenizer.texts_to_token_ids([prompt_text,text])
 if not prompt or not target:raise ValueError('Both reference text and target text must produce tokens')
 duration=int(torch.ceil(torch.tensor(frames,dtype=torch.float32)*len(target)/len(prompt)).item())
 workload={'batch':manifest['batch'],'prompt_frames':375,'target_frames':duration,'total_frames':375+duration,'joint_tokens':len(prompt)+len(target),'padded_tokens':len(prompt)+len(target)+1,'steps':8,'t_shift':.5,'guidance':1.,'feat_scale':.1}
 check_workload(manifest,workload)
 output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
 save_file({'prompt_mel':mel.unsqueeze(0),'token_ids':torch.tensor([prompt+target+[tokenizer.pad_id]],dtype=torch.int64),'prompt_rms':rms.reshape(1)},str(output))
 output.with_suffix('.workload.json').write_text(json.dumps(workload,indent=2)+'\n')
 return workload
