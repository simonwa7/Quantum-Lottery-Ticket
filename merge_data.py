import sys
import json

with open(sys.argv[1], "r") as f:
    data1 = json.loads(f.read())
f.close()
with open(sys.argv[2], "r") as f:
    data2 = json.loads(f.read())
f.close()

data = data1 | data2

with open(sys.argv[2], "w") as f:
    f.write(json.dumps(data))
f.close()
