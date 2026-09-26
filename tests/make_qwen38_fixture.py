#!/usr/bin/env python3
import struct
from pathlib import Path

MAGIC = 0x46554747
STRING = 8
U32 = 4

def s(x: str) -> bytes:
    b=x.encode()
    return struct.pack('<Q',len(b))+b

def kv_string(k,v):
    return s(k)+struct.pack('<I',STRING)+s(v)

def kv_u32(k,v):
    return s(k)+struct.pack('<I',U32)+struct.pack('<I',v)

def write(path: Path):
    kvs=[
        kv_string('general.architecture','qwen4exp'),
        kv_string('general.name','Qwen3.8-Flash-Next test'),
        kv_u32('qwen4exp.block_count',48),
        kv_u32('qwen4exp.embedding_length',2560),
        kv_u32('qwen4exp.expert_count',512),
        kv_u32('qwen4exp.expert_used_count',10),
        kv_u32('qwen4exp.nextn_predict_layers',1),
    ]
    body=struct.pack('<IIQQ',MAGIC,3,0,len(kvs))+b''.join(kvs)
    body += b'\0' * ((32-len(body)%32)%32)
    path.write_bytes(body)

root=Path(__file__).resolve().parent/'qwen38-fixture'
root.mkdir(exist_ok=True)
write(root/'Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-00001-of-00002.gguf')
(root/'Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-00002-of-00002.gguf').write_bytes(b'x'*17)
