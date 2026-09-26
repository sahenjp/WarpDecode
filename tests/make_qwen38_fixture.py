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

ARRAY = 9
I32 = 5
U64 = 10

def kv_array_i32(k, values):
    return (
        s(k)
        + struct.pack('<I', ARRAY)
        + struct.pack('<IQ', I32, len(values))
        + b''.join(struct.pack('<i', v) for v in values)
    )

def kv_array_u64(k, values):
    return (
        s(k)
        + struct.pack('<I', ARRAY)
        + struct.pack('<IQ', U64, len(values))
        + b''.join(struct.pack('<Q', v) for v in values)
    )

def write(path: Path):
    kvs=[
        kv_string('general.architecture','qwen4exp'),
        kv_string('general.name','Qwen3.8-Flash-Next test'),
        kv_u32('qwen4exp.block_count',49),
        kv_u32('qwen4exp.embedding_length',2560),
        kv_u32('qwen4exp.attention.head_count',24),
        kv_u32('qwen4exp.attention.head_count_kv',2),
        kv_u32('qwen4exp.attention.key_length',256),
        kv_u32('qwen4exp.attention.value_length',256),
        kv_u32('qwen4exp.expert_count',512),
        kv_u32('qwen4exp.expert_used_count',10),
        kv_u32('qwen4exp.expert_feed_forward_length',640),
        kv_u32('qwen4exp.expert_shared_feed_forward_length',640),
        kv_u32('qwen4exp.ssm.conv_kernel',4),
        kv_u32('qwen4exp.ssm.state_size',128),
        kv_u32('qwen4exp.ssm.group_count',16),
        kv_u32('qwen4exp.ssm.time_step_rank',48),
        kv_u32('qwen4exp.ssm.inner_size',6144),
        kv_u32('qwen4exp.full_attention_interval',4),
        kv_u32('qwen4exp.hyper_connection.count',4),
        kv_u32('qwen4exp.hyper_connection.low_rank',320),
        kv_u32('qwen4exp.attention.indexer.head_count',4),
        kv_u32('qwen4exp.attention.indexer.key_length',128),
        kv_u32('qwen4exp.attention.indexer.top_k',2048),
        kv_u32('qwen4exp.nextn_predict_layers',1),
        kv_array_i32('qwen4exp.attention.compress_ratios', [0, 0, 0, 8] * 12 + [0]),
        kv_array_i32('qwen4exp.ple.layers', [0]),
        kv_u32('qwen4exp.ple.ngram_size',3),
        kv_u32('qwen4exp.ple.heads_per_ngram',8),
        kv_u32('qwen4exp.ple.conv_kernel',4),
        kv_u32('qwen4exp.embedding_length_per_layer_input',160),
        kv_array_u64('qwen4exp.ple.layer_multipliers', [3, 5, 7]),
        kv_array_u64('qwen4exp.ple.head_offsets', list(range(0, 160, 10))),
        kv_array_u64('qwen4exp.ple.head_vocab_sizes', [10] * 16),
    ]
    body=struct.pack('<IIQQ',MAGIC,3,0,len(kvs))+b''.join(kvs)
    body += b'\0' * ((32-len(body)%32)%32)
    path.write_bytes(body)

root=Path(__file__).resolve().parent/'qwen38-fixture'
root.mkdir(exist_ok=True)
write(root/'Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-00001-of-00002.gguf')
(root/'Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-00002-of-00002.gguf').write_bytes(b'x'*17)
