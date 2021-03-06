from jaratoolbox.test.nick.ephysExperiments import ephys_experiment_v2 as ee2
reload(ee2) 

sessionTypes = {'nb':'noiseBurst',
                'lp':'laserPulse',
                'lt':'laserTrain',
                'tc':'tuningCurve',
                'bf':'bestFreq',
                '3p':'3mWpulse',
                '1p':'1mWpulse'} 

today = ee2.RecordingDay(animalName = 'd1pi001', date = '2015-07-01', experimenter = 'lan')

site1 = ee2.RecordingSite(today, depth = 2612,goodTetrodes = [6])
site1.add_session('18-06-54',None, sessionTypes['lp'])
site1.add_session('18-12-05',None, sessionTypes['nb'])
site1.add_session('18-18-47', 'a', sessionTypes['tc'])                     
site1.add_session('18-36-00', None, sessionTypes['3p'])            
         

site2 = ee2. RecordingSite(today, depth = 2660,goodTetrodes = [6])
site2.add_session('18-42-37', None, sessionTypes['nb'])
site2.add_session('18-45-15', None, sessionTypes['lp'])
site2.add_session('18-48-18', None, sessionTypes['lt'])
site2.add_session('18-51-53', None, sessionTypes['3p'])
site2.add_session('18-55-15', 'b', sessionTypes['tc'])
                             

site3 = ee2.RecordingSite(today, depth = 2710,goodTetrodes = [6])
site3.add_session('19-09-42', None, sessionTypes['nb'])
site3.add_session('19-13-36', None, sessionTypes['lp'])
site3.add_session('19-16-32', None, sessionTypes['lt'])
site3.add_session('19-23-35', None, sessionTypes['1p'])  
site3.add_session('19-30-41', 'c', sessionTypes['tc'])

cluster4 = site3.add_cluster(clusterNumber=4, tetrode=6, soundResponsive=True, laserPulseResponse=False, followsLaserTrain=False, comments='good cell')
cluster6 = site3.add_cluster(clusterNumber=6, tetrode=6, soundResponsive=True, laserPulseResponse=False, followsLaserTrain=False, comments = 'good cell')
cluster10 = site3.add_cluster(clusterNumber=10, tetrode=6, soundResponsive=True, laserPulseResponse=True, followsLaserTrain=True, comments='good cell')
cluster12 = site3.add_cluster(clusterNumber=12, tetrode=6, soundResponsive=True, laserPulseResponse=True,followsLaserTrain=True, comments='inhibited by laser')                  


site4 = ee2.RecordingSite(today, depth = 2800,  goodTetrodes = [3, 6])
site4.add_session('19-49-07', None, sessionTypes['lp'])
site4.add_session('19-52-08', None, sessionTypes['nb'])
site4.add_session('19-54-43', None, sessionTypes['lt'])
site4.add_session('19-58-34', 'd', sessionTypes['tc'])
site4.add_session('20-11-17', None, sessionTypes['3p'])
site4.add_session('20-14-20', None, sessionTypes['1p'])                     


site5 = ee2.RecordingSite(today, depth = 2900, goodTetrodes = [3, 6])
site5.add_session('20-25-09', None, sessionTypes['lp'])
site5.add_session('20-28-03', None, sessionTypes['lt'])
site5.add_session('20-31-44', None, sessionTypes['nb'])
site5.add_session('20-34-08', None, sessionTypes['3p']) 
site5.add_session('20-39-59', 'e', sessionTypes['tc'])

site6 = ee2.RecordingSite(today, depth = 3025,goodTetrodes = [6])
site6.add_session('21-01-24', None, sessionTypes['nb'])
site6.add_session('21-03-51', None, sessionTypes['lp'])
site6.add_session('21-06-10', None, sessionTypes['lt'])
site6.add_session('21-09-42', None, sessionTypes['1p'])
site6.add_session('21-12-07', 'f', sessionTypes['tc'])

from jaratoolbox.test.lan.Ephys import laserTCanalysis_added_clu_report as laserTC
reload(laserTC)
for indSite, site in enumerate(today.siteList):
    laserTC.laser_tc_analysis(site, indSite+1)
