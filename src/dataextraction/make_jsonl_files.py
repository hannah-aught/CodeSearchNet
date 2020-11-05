import pickle
import json
import gzip

for i in range(31, 32):
    base_file_path = f'/data2/ainedonn/ecosystem_csndata/aaeg/java_definitions_{i}'
    with open(base_file_path + '.pkl','rb') as f:
        defs = pickle.load(f)
    with gzip.open(base_file_path + '.jsonl.gz', 'wb') as f:
        written = 0
        print(f'starting {base_file_path}.jsonl.gz: {len(defs)} definitions')
        for d in defs:
            for x in d:
                f.write(bytes(json.dumps(x),'utf-8'))
                f.write(b'\n')
            print(f'repos written: {written}/{len(defs)}', end='\r')
            written += 1

        print('wrote data in {}.pkl to {}.jsonl.gz'.format(base_file_path, base_file_path)) 
