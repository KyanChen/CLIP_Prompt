import os


for i in range(0, 127):
    os.system(
        f'wget https://huggingface.co/datasets/laion/laion2B-en/resolve/main/part-${i:5d}-5114fd87-297e-42b0-9d11-50f1df323dfa-c000.snappy.parquet')

'wget https://huggingface.co/datasets/laion/laion2B-en/resolve/main/part-$00001-5114fd87-297e-42b0-9d11-50f1df323dfa-c000.snappy.parquet'