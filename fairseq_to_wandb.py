import json
import sys
import wandb
wandb.init(project="softadam-fairseq", entity="sourceress")

for line in sys.stdin:
    if line.startswith("Namespace("):
        namespace_dict = "dict" + line[9:-1]
        d = eval(namespace_dict)
        print("Config: {}".format(d))
        wandb.config.update(d)
    elif line.startswith("{"):
        d = json.loads(line.strip())
        d = {k: float(v) for k,v in d.items()}
        print(d)
        wandb.log(d)
