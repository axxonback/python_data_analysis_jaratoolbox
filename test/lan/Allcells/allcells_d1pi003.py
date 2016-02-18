'''
List of all isolated units from d1pi003, with cluster quality added.
Lan Guo 2016-01-11
'''
#using CellDatabase that contains laserSession for evaluating laser responsiveness

from jaratoolbox.test.lan.Ephys import celldatabase_quality_laser as celldatabase
reload(celldatabase)


eSession = celldatabase.EphysSessionInfo  # Shorter name to simplify code


cellDB = celldatabase.CellDatabase()

oneES = eSession(animalName='d1pi003',
                 ephysSession = '2016-01-25_15-22-06',
                 laserSession = '2016-01-25_15-03-28',
                 clustersEachTetrode = {1:range(1,13),2:range(1,13),3:range(1,13),4:range(1,13),5:range(1,13),6:range(1,13),7:range(1,13),8:range(1,13)},
                 behavSession = '20160125a',
                 clusterQuality = {1:[],2:[],3:[],4:[],5:[0,0,0,0,0,0,0,0,0,0,0,0],6:[3,3,3,3,3,3,3,3,3,3,3,3],7:[3,3,3,3,3,3,3,3,3,3,3,3],8:[3,7,3,7,3,3,3,6,2,3,2,7]})


oneES = eSession(animalName='d1pi003',
                 ephysSession = '2016-01-26_13-35-47',
                 laserSession = '2016-01-26_13-14-13',
                 clustersEachTetrode = {1:range(1,13),2:range(1,13),3:range(1,13),4:range(1,13),5:range(1,13),6:range(1,13),7:range(1,13),8:range(1,13)},
                 behavSession = '20160126a',
                 clusterQuality = {1:[3,4,4,2,4,1,4,4,4,1,4,0],2:[3,0,0,0,0,0,0,0,0,0,0,0],3:[3,1,1,1,4,1,3,1,1,1,1,4],4:[3,1,1,1,2,1,1,3,1,1,4,1],5:[3,3,3,3,3,2,3,3,2,3,5,2],6:[3,7,4,2,4,3,6,6,2,2,6,4],7:[3,4,4,3,3,3,3,2,1,1,1,2],8:[3,4,3,4,2,4,3,2,5,3,3,4]})
cellDB.append_session(oneES)


oneES = eSession(animalName='d1pi003',
                 ephysSession = '2016-01-27_13-27-38',
                 laserSession = '2016-01-27_13-13-22',
                 clustersEachTetrode = {1:range(1,13),2:range(1,13),3:range(1,13),4:range(1,13),5:range(1,13),6:range(1,13),7:range(1,13),8:range(1,13)},
                 behavSession = '20160127a',
                 clusterQuality = {1:[3,1,4,4,4,2,1,1,1,4,1,1],2:[3,0,0,0,0,0,0,0,0,0,0,0],3:[3,4,1,1,1,1,1,1,4,4,1,1],4:[3,1,1,1,1,1,1,1,1,1,1,1],5:[3,2,1,4,2,2,1,3,2,1,3,1],6:[3,2,1,2,5,1,2,3,6,2,6,2],7:[3,1,3,1,1,1,1,1,3,1,1,1],8:[3,3,4,5,3,1,5,3,2,3,5,3]})
cellDB.append_session(oneES)


oneES = eSession(animalName='d1pi003',
                 ephysSession = '2016-01-28_10-06-46',
                 laserSession = '2016-01-28_09-49-32',
                 clustersEachTetrode = {1:range(1,13),2:range(1,13),3:range(1,13),4:range(1,13),5:range(1,13),6:range(1,13),7:range(1,13),8:range(1,13)},
                 behavSession = '20160128a',
                 clusterQuality = {1:[3,4,1,1,4,4,4,1,1,4,4,4],2:[3,0,0,0,0,0,0,0,0,0,0,0],3:[3,1,1,4,1,4,1,4,1,1,4,1],4:[3,1,4,4,1,1,1,1,4,4,1,1],5:[3,2,3,3,4,4,4,4,4,4,3,4],6:[3,2,1,4,1,1,1,2,1,3,2,1,1],7:[3,4,1,1,1,1,2,1,1,2,1,3],8:[3,3,4,3,5,3,3,5,5,3,3,3]})
cellDB.append_session(oneES)


oneES = eSession(animalName='d1pi003',
                 ephysSession = '2016-01-29_11-32-32',
                 laserSession = '2016-01-29_13-03-48',
                 clustersEachTetrode = {1:range(1,13),2:range(1,13),3:range(1,13),4:range(1,13),5:range(1,13),6:range(1,13),7:range(1,13),8:range(1,13)},
                 behavSession = '20160129a',
                 clusterQuality = {1:[3,1,1,1,1,1,1,1,1,4,1,4],2:[3,0,0,0,0,0,0,0,0,0,0,0],3:[3,4,1,1,4,3,1,4,4,1,4,0],4:[3,1,4,4,1,1,1,1,4,4,1,1],5:[3,2,3,3,4,4,4,4,4,4,3,4],6:[3,2,1,4,1,1,1,2,1,3,2,1,1],7:[3,4,1,1,1,1,2,1,1,2,1,3],8:[3,3,4,3,5,3,3,5,5,3,3,3]})
cellDB.append_session(oneES)


oneES = eSession(animalName='d1pi003',
                 ephysSession = '2016-02-03_14-45-23',
                 laserSession = '2016-02-03_16-16-58',
                 clustersEachTetrode = {1:range(1,13),2:range(1,13),3:range(1,13),4:range(1,13),5:range(1,13),6:range(1,13),7:range(1,13),8:range(1,13)},
                 behavSession = '20160203a',
                 clusterQuality = {1:[3,4,1,1,1,3,4,1,1,1,1,0],2:[3,3,3,3,3,3,3,7,3,4,3,4],3:[3,4,4,4,3,3,1,1,4,4,4,4],4:[3,1,2,3,1,1,1,1,1,1,1,1],5:[3,3,3,3,3,3,4,4,1,3,3,3],6:[3,4,4,4,3,3,1,1,1,1,3,4],7:[3,1,4,1,4,3,4,4,1,1,1,1],8:[0,0,0,0,0,0,0,0,0,0,0,0]}) #Did not have behav graphs since behav has more trials than ephys
cellDB.append_session(oneES)


oneES = eSession(animalName='d1pi003',
                 ephysSession = '2016-02-04_14-21-32',
                 laserSession = '2016-02-04_15-52-19',
                 clustersEachTetrode = {1:range(1,13),2:range(1,13),3:range(1,13),4:range(1,13),5:range(1,13),6:range(1,13),7:range(1,13),8:range(1,13)},
                 behavSession = '20160204a',
                 clusterQuality = {1:[3,1,1,1,1,1,3,1,4,1,1,1],2:[3,3,3,3,1,3,3,3,3,3,3,4],3:[3,3,3,3,3,3,4,3,3,2,6,3],4:[3,1,1,1,1,1,3,1,1,1,1,1],5:[3,3,1,4,1,3,3,3,3,3,4,1],6:[3,1,3,4,4,1,4,3,7,3,4,3],7:[3,3,1,4,1,1,4,1,4,1,1,4],8:[3,0,0,0,0,0,0,0,0,0,0,0]}) 
cellDB.append_session(oneES)


oneES = eSession(animalName='d1pi003',
                 ephysSession = '2016-02-05_15-03-04',
                 laserSession = '2016-02-05_16-35-08',
                 clustersEachTetrode = {1:range(1,13),2:range(1,13),3:range(1,13),4:range(1,13),5:range(1,13),6:range(1,13),7:range(1,13),8:range(1,13)},
                 behavSession = '20160205a',
                 clusterQuality = {1:[3,1,3,1,1,1,1,2,1,4,1,3],2:[3,3,3,3,3,3,3,3,3,3,3,3],3:[3,3,3,3,3,3,2,3,3,3,4,3],4:[3,3,1,1,1,1,1,1,1,1,1,1],5:[3,7,6,6,1,6,3,7,3,7,3,6],6:[3,3,3,1,1,2,1,1,4,1,1,1],7:[3,1,3,7,6,7,6,2,1,1,2,1],8:[3,0,0,0,0,0,0,0,0,0,0,0]})
cellDB.append_session(oneES)


oneES = eSession(animalName='d1pi003',
                 ephysSession = '2016-02-06_16-21-02',
                 laserSession = '2016-02-06_16-36-09',
                 clustersEachTetrode = {1:range(1,13),2:range(1,13),3:range(1,13),4:range(1,13),5:range(1,13),6:range(1,13),7:range(1,13),8:range(1,13)},
                 behavSession = '20160206a',
                 clusterQuality = {1:[3,1,1,1,3,1,1,4,1,1,4,4],2:[3,3,3,3,3,3,3,3,3,3,3,0],3:[3,4,3,3,3,3,2,3,3,3,4,3],4:[3,1,1,1,1,1,1,1,1,1,1,1],5:[3,7,3,7,3,4,3,4,4,3,4,3],6:[3,1,6,3,1,6,1,1,2,7,1,0],7:[3,7,6,6,6,6,3,6,4,1,4,6],8:[0,0,0,0,0,0,0,0,0,0,0,0]}) 
cellDB.append_session(oneES)


oneES = eSession(animalName='d1pi003',
                 ephysSession = '2016-02-08_15-26-48',
                 laserSession = '2016-02-08_15-40-40',
                 clustersEachTetrode = {1:range(1,13),2:range(1,13),3:range(1,13),4:range(1,13),5:range(1,13),6:range(1,13),7:range(1,13),8:range(1,13)},
                 behavSession = '20160208a',
                 clusterQuality = {1:[3,1,1,1,1,1,2,1,1,1,3,1],2:[3,3,3,3,3,3,3,3,3,3,3,0],3:[3,4,1,1,3,3,3,4,1,3,3,3],4:[3,1,1,1,1,1,1,1,4,1,1,1],5:[3,3,4,3,3,3,4,2,3,3,3,3],6:[3,4,1,1,2,4,1,4,3,6,6,4],7:[3,1,1,1,4,1,2,1,1,3,1,1],8:[0,0,0,0,0,0,0,0,0,0,0,0]})
cellDB.append_session(oneES)


oneES = eSession(animalName='d1pi003',
                 ephysSession = '2016-02-09_16-17-41',
                 laserSession = '2016-02-09_16-00-21',
                 clustersEachTetrode = {1:range(1,13),2:range(1,13),3:range(1,13),4:range(1,13),5:range(1,13),6:range(1,13),7:range(1,13),8:range(1,13)},
                 behavSession = '20160209a',
                 clusterQuality = {1:[3,4,4,4,7,1,1,3,1,4,1,4],2:[3,3,3,3,3,3,3,3,3,3,5,3],3:[3,3,4,3,1,4,4,1,1,3,3,3],4:[3,1,1,1,1,1,1,1,1,1,4,1],5:[3,3,3,4,1,3,6,3,3,4,4,6],6:[3,1,3,4,4,4,3,1,6,4,4,4],7:[3,1,1,1,1,4,1,1,7,1,1,4],8:[0,0,0,0,0,0,0,0,0,0,0,0]})
cellDB.append_session(oneES)


oneES = eSession(animalName='d1pi003',
                 ephysSession = '2016-02-10_15-25-03',
                 laserSession = '2016-02-10_15-08-57',
                 clustersEachTetrode = {1:range(1,13),2:range(1,13),3:range(1,13),4:range(1,13),5:range(1,13),6:range(1,13),7:range(1,13),8:range(1,13)},
                 behavSession = '20160210a',
                 clusterQuality = {1:[3,1,1,4,3,1,6,6,1,3,6,1],2:[3,3,3,3,3,3,3,3,3,3,5,3],3:[3,4,3,3,3,4,4,3,3,3,4,3],4:[3,3,4,1,1,1,1,1,1,1,3,1],5:[3,3,6,3,6,6,3,6,3,3,3,3],6:[3,3,3,7,3,1,1,2,4,1,1,4],7:[3,1,1,1,2,3,3,2,1,1,1,1],8:[0,0,0,0,0,0,0,0,0,0,0,0]})
cellDB.append_session(oneES)

oneES = eSession(animalName='d1pi003',
                 ephysSession = '2016-02-11_15-07-45',
                 laserSession = '2016-02-11_14-40-51',
                 clustersEachTetrode = {1:range(1,13),2:range(1,13),3:range(1,13),4:range(1,13),5:range(1,13),6:range(1,13),7:range(1,13),8:range(1,13)},
                 behavSession = '20160211a',
                 clusterQuality = {1:[3,1,1,4,1,1,4,7,1,7,6,6],2:[3,3,3,3,3,3,3,3,3,7,4,3],3:[3,3,3,3,3,4,4,4,4,1,4,3],4:[3,1,1,1,1,1,1,1,1,1,1,1],5:[3,4,4,2,4,3,3,3,4,3,7,6],6:[3,1,1,1,2,1,4,1,4,1,4,4],7:[3,2,2,4,6,1,6,6,1,4,6,1],8:[0,0,0,0,0,0,0,0,0,0,0,0]})
cellDB.append_session(oneES)


oneES = eSession(animalName='d1pi003',
                 ephysSession = '2016-02-12_14-34-32',
                 laserSession = '2016-02-12_14-17-01',
                 clustersEachTetrode = {1:range(1,13),2:range(1,13),3:range(1,13),4:range(1,13),5:range(1,13),6:range(1,13),7:range(1,13),8:range(1,13)},
                 behavSession = '20160212a',
                 clusterQuality = {1:[3,1,4,1,1,1,4,4,1,4,7,3],2:[3,3,3,3,3,1,3,3,3,3,2,3],3:[3,3,2,2,1,3,3,4,2,3,4,4],4:[3,1,1,1,1,1,1,1,1,1,1,1],5:[3,4,3,2,4,2,2,4,3,4,2,2],6:[3,1,1,5,4,4,1,4,1,1,1,7],7:[3,2,1,6,6,1,1,1,1,3,4,7],8:[0,0,0,0,0,0,0,0,0,0,0,0]})
cellDB.append_session(oneES)


oneES = eSession(animalName='d1pi003',
                 ephysSession = '2016-02-13_17-36-49',
                 laserSession = '2016-02-13_17-49-46',
                 clustersEachTetrode = {1:range(1,13),2:range(1,13),3:range(1,13),4:range(1,13),5:range(1,13),6:range(1,13),7:range(1,13),8:range(1,13)},
                 behavSession = '20160213a',
                 clusterQuality = {1:[3,1,1,6,2,3,1,1,1,2,1,1],2:[3,2,2,4,4,3,3,3,3,3,2,1],3:[3,3,2,2,2,3,1,2,3,3,2,4],4:[3,1,1,1,1,1,2,1,3,1,4,1],5:[3,2,2,4,4,2,2,3,2,3,4,4],6:[3,1,1,1,1,4,1,1,7,1,1,0],7:[3,1,2,2,1,1,2,1,1,1,2,1],8:[0,0,0,0,0,0,0,0,0,0,0,0]})
cellDB.append_session(oneES)


oneES = eSession(animalName='d1pi003',
                 ephysSession = '2016-02-15_14-58-06',
                 laserSession = '2016-02-15_15-10-34',
                 clustersEachTetrode = {1:range(1,13),2:range(1,13),3:range(1,13),4:range(1,13),5:range(1,13),6:range(1,13),7:range(1,13),8:range(1,13)},
                 behavSession = '20160215a',
                 clusterQuality = {1:[3,1,1,4,4,1,1,1,1,3,4,2],2:[3,2,2,4,4,4,2,1,2,3,3,1],3:[3,3,3,2,3,2,4,4,4,2,2,2],4:[3,1,1,2,1,1,4,4,1,1,1,1],5:[3,7,2,3,3,7,4,4,4,2,2,0],6:[3,4,1,1,3,1,2,4,1,1,1,4],7:[3,4,1,1,2,3,1,1,1,4,1,1],8:[0,0,0,0,0,0,0,0,0,0,0,0]})
cellDB.append_session(oneES)


oneES = eSession(animalName='d1pi003',
                 ephysSession = '2016-02-16_',
                 laserSession = '2016-02-16_',
                 clustersEachTetrode = {1:range(1,13),2:range(1,13),3:range(1,13),4:range(1,13),5:range(1,13),6:range(1,13),7:range(1,13),8:range(1,13)},
                 behavSession = '20160216a',
                 clusterQuality = {1:[3,4,1,3,3,3,2,1,3,3,3,6],2:[3,3,4,3,3,3,3,2,3,3,3,3],3:[3,3,3,3,3,3,3,3,3,2,3,4],4:[3,3,1,4,1,4,3,1,1,1,4,1],5:[3,3,3,3,6,4,3,3,3,4,4,3],6:[3,3,3,4,3,3,3,4,3,1,3,6],7:[3,3,3,1,3,4,2,3,3,1,1,0],8:[0,0,0,0,0,0,0,0,0,0,0,0]})
cellDB.append_session(oneES)
