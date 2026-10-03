"""Explicit CPU compatibility defaults for the maintained branch."""
import os,sys,json
import torch,numpy

def old_compatible():
    value=os.environ.get('TRACER_OLD_COMPATIBLE','1')
    if value not in {'0','1'}:raise ValueError('TRACER_OLD_COMPATIBLE must be 0 or 1')
    return value=='1'

def get_device():
    device=torch.device(os.environ.get('TRACER_DEVICE','cpu'))
    if old_compatible() and device.type!='cpu':raise ValueError('Old-compatible mode requires CPU; set TRACER_OLD_COMPATIBLE=0 for standard GPU math')
    return device

def metadata():
    return {'python':sys.version,'torch':str(torch.__version__),'numpy':str(numpy.__version__),'device':str(get_device()),'old_compatible':old_compatible()}

def announce():
    print('[TRACER runtime] '+json.dumps(metadata()),flush=True)
