import pytest
import torch
from inspark_infer.build.nvfp4_graph import MatrixWeightOp

@pytest.mark.parametrize('layout',['channel_tap','tap_channel'])
@pytest.mark.parametrize('transpose,stride,padding,dilation,outpad',[
    (False,1,1,1,0),(False,2,2,2,0),(True,4,2,1,0),(True,2,1,2,1),(True,2,4,1,0)])
def test_convolution_boundaries_and_overlap(layout,transpose,stride,padding,dilation,outpad):
    torch.manual_seed(41)
    args={'stride':stride,'padding':padding,'dilation':dilation}
    if transpose:args['output_padding']=outpad
    layer=(torch.nn.ConvTranspose1d if transpose else torch.nn.Conv1d)(3,5,3,**args).double()
    op=MatrixWeightOp(layer,dtype=torch.float64,conv_layout=layout).double()
    # Impulses at both boundaries exercise crop, zero insertion and overlapping sums.
    x=torch.zeros(2,3,13,dtype=torch.float64);x[:,:,0]=1;x[:,:,-1]=-1;x[1,:,6]=2
    ref=layer(x);actual=op(x)
    assert ref.shape==actual.shape
    torch.testing.assert_close(actual.double(),ref,atol=2e-7,rtol=2e-7)
