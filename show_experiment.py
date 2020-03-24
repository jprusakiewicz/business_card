import pandas as pd
import json
results = pd.read_csv("experiment_out.csv")
#print(results)
#for column in results:
    #print(results[column])
dictionary = {}
for row in results.values:
    l = [float(a) if a != 'None' else 0 for a in row[1:] ]
    m = pd.Series(l)
    print(m)
    idx = m.idxmax()
    #print(row[0], (idx+1)*50)
    dictionary[row[0]] = (idx+1)*50
print(dictionary)
json = json.dumps(dictionary)
f = open("experiment_best_performance.json","w")
f.write(json)
f.close()