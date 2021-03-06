from jaratoolbox import settings
from jaratoolbox import spikesorting
from jaratoolbox import loadopenephys
from jaratoolbox import spikesanalysis
from jaratoolbox.spikesorting import *
from pylab import *
import numpy as np
reload(spikesorting)
import os

'''The goal is to load many sessions, put the spikes from all of the sessions into a single structure in some kind of organized fashion, and then cluster the whole structure at once so that we can compare a cell across multiple sessions. 
'''

class MultipleSessionsToCluster(spikesorting.TetrodeToCluster):

    '''
    This is sort of a hybrid frankenstein class, somewhere in between the TetrodeToCluster class in spikesorting and the Dataspikes class in loadopenephys. 
    '''

    def __init__(self, animalName, sessionList, tetrode, analysisDate):
        self.animalName = animalName
        self.sessionList = sessionList
        self.SAMPLING_RATE = 30000.0
        self.tetrode = tetrode
        self.dataTT = None
        self.recordingNumber = np.array([])
        self.samples = None
        self.timestamps = None
        self.clusters = None
        self.nSpikes = None
        self.analysisDate = analysisDate
        self.clustersDir = os.path.join(settings.EPHYS_PATH,self.animalName,'multisession_{}'.format(self.analysisDate))
        self.fetFilename = os.path.join(self.clustersDir,'Tetrode%d.fet.1'%self.tetrode)

        self.report = None
        self.reportFileName = '{0}.png'.format(self.tetrode)
        
        self.featureNames = ['peak','valley','energy']
        self.nFeatures = len(self.featureNames)
        self.featureValues = None

        self.process = None
        
    def load_all_waveforms(self):
        for ind, session in enumerate(self.sessionList):
            if session: #This is a fix for when some sessions are 'None'
                ephysDir = os.path.join(settings.EPHYS_PATH, self.animalName, session)
                spikeFile = os.path.join(ephysDir, 'Tetrode{0}.spikes'.format(self.tetrode))
                dataSpkObj = loadopenephys.DataSpikes(spikeFile)
                numSpikes = dataSpkObj.nRecords
            
                #Add the ind to the vector of zeros, indicates which recording this is from. 
                sessionVector = dataSpkObj.recordingNumber+ind

                #Have to 
                samplesThisSession = dataSpkObj.samples.astype(float)-2**15# FIXME: this is specific to OpenEphys
                samplesThisSession = (1000.0/dataSpkObj.gain[0,0]) * samplesThisSession
                timestampsThisSession = dataSpkObj.timestamps/self.SAMPLING_RATE
            
                #Set the values when working with the first session, then append for the other sessions. 
                if ind==0:
                    self.samples = samplesThisSession
                    self.timestamps = timestampsThisSession
                    self.recordingNumber = sessionVector
                else:
                    self.samples = np.concatenate([self.samples, samplesThisSession])
                    self.timestamps = np.concatenate([self.timestamps, timestampsThisSession])
                    self.recordingNumber = np.concatenate([self.recordingNumber, sessionVector])
            
        self.nSpikes = len(self.timestamps)

    def create_multisession_fet_files(self):
        if not os.path.exists(self.clustersDir):
            print 'Creating clusters directory: %s'%(self.clustersDir)
            os.makedirs(self.clustersDir)
        if self.samples is None:
            self.load_waveforms()
        self.featureValues = calculate_features(self.samples,self.featureNames)
        write_fet_file(self.fetFilename, self.featureValues)
        
    def set_clusters_from_file(self):
        clusterFile = os.path.join(self.clustersDir,'Tetrode%d.clu.1'%self.tetrode)
        self.clusters = np.fromfile(clusterFile,dtype='int32',sep=' ')[1:]

    def save_multisession_report(self):
        if self.clusters == None:
            self.set_clusters_from_file()
        figTitle = 'Multisession Report TT{}'.format(self.tetrode)
        self.report = MultiSessionClusterReport(self.samples, self.timestamps, self.clusters,
                                            outputDir=self.clustersDir,
                                            filename=self.reportFileName,figtitle=figTitle,
                                                showfig=True)

        
class MultiSessionClusterReport(object):
    '''
    Need to finish reports when more than nrows<clusters.
    '''
    def __init__(self,samples, timestamps, clusters ,outputDir=None,filename=None,showfig=True,figtitle='',nrows=12):
        self.timestamps = timestamps
        self.samples = samples
        self.nSpikes = len(self.timestamps)
        self.clusters = clusters
        self.clustersList = np.unique(self.clusters)
        self.nClusters = len(self.clustersList)
        self.nRows = nrows
        self.nPages = self.nClusters//(self.nRows+1)+1
        self.spikesEachCluster = [] # Bool
        self.find_spikes_each_cluster()
        #self.fig = plt.figure(fignum)
        self.fig = None
        self.nPages = 0
        self.figTitle = figtitle
        
        self.plot_report(showfig=showfig)
        if outputDir is not None:
            self.save_report(outputDir,filename)
    def __str__(self):
        return '%d clusters'%(self.nClusters)
    def find_spikes_each_cluster(self):
        self.spikesEachCluster = np.empty((self.nClusters,self.nSpikes),dtype=bool)
        for indc,clusterID in enumerate(self.clustersList):
            self.spikesEachCluster[indc,:] = (self.clusters==clusterID)
    def plot_report(self,showfig=False):
        print 'Plotting report...'
        #plt.figure(self.fig)
        self.fig = plt.gcf()
        self.fig.clf()
        self.fig.set_facecolor('w')
        nCols = 3
        nRows = self.nRows
        #for indc,clusterID in enumerate(self.clustersList[:3]):
        for indc,clusterID in enumerate(self.clustersList):
            #print('Preparing cluster %d'%clusterID)
            if (indc+1)>self.nRows:
                print 'WARNING! This cluster was ignore (more clusters than rows)'
                continue
            tsThisCluster = self.timestamps[self.spikesEachCluster[indc,:]]
            wavesThisCluster = self.samples[self.spikesEachCluster[indc,:],:,:]
            # -- Plot ISI histogram --
            plt.subplot(self.nRows,nCols,indc*nCols+1)
            plot_isi_loghist(tsThisCluster)
            if indc<(self.nClusters-1): #indc<2:#
                plt.xlabel('')
                plt.gca().set_xticklabels('')
            plt.ylabel('c%d'%clusterID,rotation=0,va='center',ha='center')
            # -- Plot events in time --
            plt.subplot(2*self.nRows,nCols,2*(indc*nCols)+6)
            plot_events_in_time(tsThisCluster)
            if indc<(self.nClusters-1): #indc<2:#
                plt.xlabel('')
                plt.gca().set_xticklabels('')
            # -- Plot projections --
            plt.subplot(2*self.nRows,nCols,2*(indc*nCols)+3)
            plot_projections(wavesThisCluster)  
            # -- Plot waveforms --
            plt.subplot(self.nRows,nCols,indc*nCols+2)
            plot_waveforms(wavesThisCluster)
        #figTitle = self.get_title()
        plt.figtext(0.5,0.92, self.figTitle,ha='center',fontweight='bold',fontsize=10)
        if showfig:
            #plt.draw()
            plt.show()
    def get_title(self):
        return ''
    def get_default_filename(self,figformat):
        return 'clusterReport.%s'%(figformat)
    def save_report(self,outputdir,filename=None,figformat=None):
        # -- Create output directory --
        if not os.path.exists(outputdir):
            print 'Creating clusters directory: %s'%(outputdir)
            os.makedirs(outputdir)
        self.fig.set_size_inches((8.5,11))
        if figformat is None:
            figformat = 'png' #'png' #'pdf' #'svg'
        if filename is None:
            filename = self.get_default_filename(figformat)
        fullFileName = os.path.join(outputdir,filename)
        print 'Saving figure to %s'%fullFileName
        self.fig.savefig(fullFileName,format=figformat)
        #plt.close(self.fig)
        ###def closefig(self):



if __name__=="__main__":
    
    from jaratoolbox.test.nick.ephysExperiments import ephys_experiment
    from jaratoolbox.test.nick.ephysExperiments.ephys_experiment import EphysExperiment
    reload(ephys_experiment)

    animalName = 'pinp003'
    #2015-06-22_18-57-35 - Tuning curve
    #2015-06-22_19-43-42 - 100msec tones, 7.5-8kHz. Good responses
    #2015-06-22_19-52-23 - 100msec laser pulses at 3mW, spikes then silence
    #2015-06-22_19-57-14 - 100msec laser pulses at 2mW
    #2015-06-22_19-59-48 - 100msec laser pulses at 1mW
    sessionList = ['2015-06-24_15-22-29', '2015-06-24_15-25-08', '2015-06-24_15-27-37', '2015-06-24_15-31-48', '2015-06-24_15-45-22']
                      
    tetrode = 6
    oneTT = MultipleSessionsToCluster(animalName,sessionList,tetrode, '20150626site1')



    oneTT.load_all_waveforms()

    clusterFile = os.path.join(oneTT.clustersDir,'Tetrode%d.clu.1'%oneTT.tetrode)
    if os.path.isfile(clusterFile):
       oneTT.set_clusters_from_file() 
    else:
        oneTT.create_multisession_fet_files()
        oneTT.run_clustering()
    oneTT.save_multisession_report()
    
    ##PLot the noise burst response for a cluster

    figure()
    cluster = 3
    clusterSpikeTimestamps = oneTT.timestamps[(oneTT.clusters==cluster) & (oneTT.recordingNumber==0)]
    

    exp = EphysExperiment('pinp003', '2015-06-24')
    
    spikeData, eventData, plotTitle = exp.get_session_ephys_data(sessionList[2], 6)
    
    eventOnsetTimes = exp.get_event_onset_times(eventData)
    
    exp.plot_raster(clusterSpikeTimestamps, eventOnsetTimes, 'laserResponse')    
    



