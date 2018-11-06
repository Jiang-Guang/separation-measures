import numpy as np
import separation_measures
import multiprocessing
import json

def work(data):
    try:
        points = np.array(data[0]).tolist()
        label = np.array(data[1]).tolist()
        index = data[2]
        abtn = separation_measures.ABTN(points, label)
        awtn = separation_measures.AWTN(points, label)
        abw = abtn*1.0 / awtn
        cal = separation_measures.CAL(points, label)
        lda = separation_measures.LDA(points, label)
        sil = separation_measures.SIL(points, label)
        cs = separation_measures.SIL(points, label)

        cdm_2 = separation_measures.CDM(points, label, k=2)
        cdm_4 = separation_measures.CDM(points, label, k=4)
        cdm_6 = separation_measures.CDM(points, label, k=6)
        cdm_8 = separation_measures.CDM(points, label, k=8)
        cdm_10 = separation_measures.CDM(points, label, k=10)

        dc_0001 = separation_measures.DC(points, label, e=0.001)
        dc_0002 = separation_measures.DC(points, label, e=0.002)
        dc_0005 = separation_measures.DC(points, label, e=0.005)
        dc_001 = separation_measures.DC(points, label, e=0.01)
        dc_005 = separation_measures.DC(points, label, e=0.05)
        dc_01 = separation_measures.DC(points, label, e=0.1)
        dc_02 = separation_measures.DC(points, label, e=0.2)

        return [abtn, awtn, abw, cal, lda, sil, cs, cdm_2, cdm_4, cdm_6, cdm_8, cdm_10, dc_0001, dc_0002, dc_0005, dc_001, dc_005, dc_01, dc_02], index
    except Exception as ex:
        print(ex)
        return [-1 for i in range(19)],index



def proc():
    source = np.load("")
    data = []
    for i in range(len(source)):
        data.append([i[0], i[1], i])
    pool = multiprocessing.Pool(processes= int(2*multiprocessing.cpu_count()/3.0))
    result = []
    for ret,index in pool.map(work, data):
        result.append([index, ret])

    with open("./result.csv", "w") as f:
        f.write("index,abtn,awtn,abw,cal,lda,sil,cs,cdm_2,cdm_4,cdm_6,cdm_8,cdm_10,dc_0001,dc_0002,dc_0005,dc_001,dc_005,dc_01,dc_02\n")
        for i in result:
            index = i[0]
            ret = i[1]
            ret = [str(j) for j in ret]
            f.write("%d,%s\n"%(index, ",".join(ret)))
if __name__ == '__main__':
    proc()
