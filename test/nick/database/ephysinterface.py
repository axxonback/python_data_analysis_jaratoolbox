'''
2015-08-01 Nick Ponvert

This module will provide a frontend for plotting data during an ephys experiment

The job of the module will be to take session names, get the data, and then pass the data to the correct plotting function
'''

from jaratoolbox.test.nick.database import dataloader
from jaratoolbox.test.nick.database import dataplotter
reload(dataplotter)
import os
from matplotlib import pyplot as plt
import numpy as np


class EphysInterface(object):

    def __init__(self,
                 animalName,
                 date,
                 experimenter,
                 defaultParadigm=None,
                 defaultElectrodes=['Electrode3', 'Electrode4', 'Electrode5', 'Electrode6'],
                 serverUser='jarauser',
                 serverName='jarahub',
                 serverBehavPathBase='/data/behavior'
                 ):

        self.animalName = animalName
        self.date = date
        self.experimenter = experimenter
        self.defaultParadigm = defaultParadigm
        self.defaultElectrodes=defaultElectrodes

        self.loader = dataloader.DataLoader('online', animalName, date, experimenter, defaultParadigm)

        self.serverUser = serverUser
        self.serverName = serverName
        self.serverBehavPathBase = serverBehavPathBase
        self.experimenter = experimenter
        self.serverBehavPath = os.path.join(self.serverBehavPathBase, self.experimenter, self.animalName)
        self.remoteBehavLocation = '{0}@{1}:{2}'.format(self.serverUser, self.serverName, self.serverBehavPath)


    #Get the behavior from jarahub
    def get_behavior(self):
        transferCommand = ['rsync', '-a', '--progress', self.remoteBehavLocation, self.localBehavPath]
        print ' '.join(transferCommand)
        subprocess.call(transferCommand)

    def plot_session_raster(self, session, electrode, cluster = None, sortArray = [], replace=0, ms=4):

        plotTitle = self.loader.get_session_filename(session)
        spikeData= self.loader.get_session_spikes(session, electrode)
        eventData = self.loader.get_session_events(session)
        eventOnsetTimes = self.loader.get_event_onset_times(eventData)
        spikeTimestamps=spikeData.timestamps

        if cluster:
            spikeTimestamps = spikeTimestamps[spikeData.clusters==cluster]

        if replace:
            plt.cla()
        else:
            plt.figure()

        dataplotter.plot_raster(spikeTimestamps, eventOnsetTimes, sortArray = sortArray, ms=ms)

        plt.show()


    def plot_array_freq_tuning(self, session, behavSuffix, replace=0, electrodes=None, timeRange=[0, 0.1]):

        if not electrodes:
            electrodes=self.defaultElectrodes

        numElectrodes = len(electrodes)
        eventData = self.loader.get_session_events(session)
        eventOnsetTimes = self.loader.get_event_onset_times(eventData)
        plotTitle = self.loader.get_session_filename(session)
        bdata=self.loader.get_session_behavior(behavSuffix)
        freqEachTrial = bdata['currentFreq']
        freqLabels = ["%.1f"%freq for freq in np.unique(freqEachTrial)/1000]

        if replace:
            fig = plt.gcf()
            plt.clf()
        else:
            fig = plt.figure()


        for ind , electrode in enumerate(electrodes):

            spikeData = self.loader.get_session_spikes(session, electrode)

            if ind == 0:
                ax = fig.add_subplot(numElectrodes,1,ind+1)
            else:
                ax = fig.add_subplot(numElectrodes,1,ind+1, sharex = fig.axes[0], sharey = fig.axes[0])

            spikeTimestamps = spikeData.timestamps
            dataplotter.one_axis_tc_or_rlf(spikeTimestamps, eventOnsetTimes, freqEachTrial, timeRange=timeRange)

            ax.set_xticks(range(len(freqLabels)))

            if ind == numElectrodes-1:
                ax.set_xticklabels(freqLabels, rotation='vertical')
                plt.xlabel('Frequency (kHz)')
            else:
                plt.setp(ax.get_xticklabels(), visible=False)


            plt.ylabel('{}'.format(electrode))


        plt.figtext(0.05, 0.5, 'Average number of spikes in range {}'.format(timeRange), rotation='vertical', va='center', ha='center')
        plt.show()



    def plot_array_raster(self, session, replace=0, sortArray=[], timeRange = [-0.5, 1], electrodes=None, ms=4):
        '''
        This is the much-improved version of a function to plot a raster for each electrode. All rasters
        will be plotted using standardized plotting code, and we will simply call the functions.
        In this case, we get the event data once, and then loop through the electrodes, getting the
        spike data and calling the plotting code for each electrode.
        '''

        if not electrodes:
            electrodes=self.defaultElectrodes

        numElectrodes = len(electrodes)
        eventData = self.loader.get_session_events(session)
        eventOnsetTimes = self.loader.get_event_onset_times(eventData)
        plotTitle = self.loader.get_session_filename(session)

        if replace:
            fig = plt.gcf()
            plt.clf()
        else:
            fig = plt.figure()


        for ind , electrode in enumerate(electrodes):

            spikeData = self.loader.get_session_spikes(session, electrode)

            if ind == 0:
                ax = fig.add_subplot(numElectrodes,1,ind+1)
            else:
                ax = fig.add_subplot(numElectrodes,1,ind+1, sharex = fig.axes[0], sharey = fig.axes[0])

            spikeTimestamps = spikeData.timestamps
            dataplotter.plot_raster(spikeTimestamps, eventOnsetTimes, sortArray=sortArray, ms=ms, timeRange = timeRange)

            if ind == 0:
                plt.title(plotTitle)

            plt.ylabel('TT {}'.format(electrode))

        plt.xlabel('time (sec)')
        plt.show()



    def plot_sorted_tuning_raster(self, session, electrode, behavSuffix, cluster = None, replace=0, timeRange = [-0.5, 1], ms = 1):
        '''
        '''
        bdata = self.loader.get_session_behavior(behavSuffix)
        plotTitle = self.loader.get_session_filename(session)
        eventData = self.loader.get_session_events(session)
        spikeData = self.loader.get_session_spikes(session, electrode)

        eventOnsetTimes = self.loader.get_event_onset_times(eventData)
        spikeTimestamps=spikeData.timestamps

        freqEachTrial = bdata['currentFreq']
        intensityEachTrial = bdata['currentIntensity']

        possibleFreq = np.unique(freqEachTrial)
        possibleIntensity = np.unique(intensityEachTrial)
        freqLabels = ['{0:.1f}'.format(freq/1000.0) for freq in possibleFreq]
        intensityLabels = ['{:.0f} dB'.format(intensity) for intensity in possibleIntensity]
        xLabel="Time from sound onset (sec)"

        plt.figure()

        dataplotter.two_axis_sorted_raster(spikeTimestamps,
                                           eventOnsetTimes,
                                           freqEachTrial,
                                           intensityEachTrial,
                                           freqLabels,
                                           intensityLabels,
                                           xLabel,
                                           plotTitle,
                                           flipFirstAxis=False,
                                           flipSecondAxis=True,
                                           timeRange=timeRange,
                                           ms=ms)

        plt.show()


    #Relies on external TC heatmap plotting functions
    def plot_session_tc_heatmap(self, session, electrode, behavSuffix, replace=True, timeRange=[0, 0.1]):
        bdata = self.loader.get_session_behavior(behavSuffix)
        plotTitle = self.loader.get_session_filename(session)
        eventData = self.loader.get_session_events(session)
        spikeData = self.loader.get_session_spikes(session, electrode)

        spikeTimestamps = spikeData.timestamps

        eventOnsetTimes = self.loader.get_event_onset_times(eventData)

        freqEachTrial = bdata['currentFreq']
        intensityEachTrial = bdata['currentIntensity']

        possibleFreq = np.unique(freqEachTrial)
        possibleIntensity = np.unique(intensityEachTrial)

        if replace:
            fig = plt.gcf()
            plt.clf()
        else:
            fig = plt.figure()


        xlabel='Frequency (kHz)'
        ylabel='Intensity (dB SPL)'

        freqLabels = ["%.1f" % freq for freq in possibleFreq/1000.0] #FIXME: This should be outside this function
        intenLabels = ['{}'.format(inten) for inten in possibleIntensity]

        dataplotter.two_axis_heatmap(spikeTimestamps,
                                     eventOnsetTimes,
                                     intensityEachTrial,
                                     freqEachTrial,
                                     intenLabels,
                                     freqLabels,
                                     xlabel,
                                     ylabel,
                                     plotTitle,
                                     flipFirstAxis=True,
                                     flipSecondAxis=False,
                                     timeRange=timeRange)

        plt.show()

    #Relies on module for clustering multiple sessions
    #Also relies on methods for plotting rasters and cluster waveforms
    def cluster_sessions_and_plot_rasters_for_each_cluster(self, ):
        pass


    def plot_LFP_tuning(self, session, channel, behavSuffix): #FIXME: Time range??
        bdata = self.loader.get_session_behavior(behavSuffix)
        plotTitle = self.loader.get_session_filename(session)
        eventData = self.loader.get_session_events(session, convertToSeconds=False)

        contData = self.loader.get_session_cont(session, channel)

        startTimestamp = contData.timestamps[0]

        eventOnsetTimes = self.loader.get_event_onset_times(eventData, diffLimit=False)

        freqEachTrial = bdata['currentFreq']
        intensityEachTrial = bdata['currentIntensity']

        possibleFreq = np.unique(freqEachTrial)
        possibleIntensity = np.unique(intensityEachTrial)

        secondsEachTrace = 0.1
        meanTraceEachSetting = np.empty((len(possibleIntensity), len(possibleFreq), secondsEachTrace*self.loader.EPHYS_SAMPLING_RATE))


        for indFreq, currentFreq in enumerate(possibleFreq):
            for indIntensity, currentIntensity in enumerate(possibleIntensity):

                #Determine which trials this setting was presented on.
                trialsThisSetting = np.flatnonzero((freqEachTrial == currentFreq) & (intensityEachTrial == currentIntensity))

                #Get the onset timestamp for each of the trials of this setting.
                timestampsThisSetting = eventOnsetTimes[trialsThisSetting]

                #Subtract the starting timestamp value to get the sample number
                sampleNumbersThisSetting = timestampsThisSetting - startTimestamp

                #Preallocate an array to store the traces for each trial on which this setting was presented.
                traces = np.empty((len(sampleNumbersThisSetting), secondsEachTrace*self.loader.EPHYS_SAMPLING_RATE))

                #Loop through all of the trials for this setting, extracting the trace after each presentation
                for indSamp, sampNumber in enumerate(sampleNumbersThisSetting):
                    trace = contData.samples[sampNumber:sampNumber + secondsEachTrace*self.loader.EPHYS_SAMPLING_RATE]
                    trace = trace - trace[0]
                    traces[indSamp, :] = trace

                #Take the mean of all of the samples for this setting, and store it according to the freq and intensity
                mean_trace = np.mean(traces, axis = 0)
                meanTraceEachSetting[indIntensity, indFreq, :] = mean_trace

        maxVoltageAllSettings = np.max(np.max(meanTraceEachSetting, axis = 2))
        minVoltageAllSettings = np.min(np.min(meanTraceEachSetting, axis = 2))

        #Plot all of the mean traces in a grid according to frequency and intensity
        for intensity in range(len(possibleIntensity)):
            #Subplot2grid plots from top to bottom, but we need to plot from bottom to top
            #on the intensity scale. So we make an array of reversed intensity indices.
            intensPlottingInds = range(len(possibleIntensity))[::-1]
            for frequency in range(len(possibleFreq)):
                plt.subplot2grid((len(possibleIntensity), len(possibleFreq)), (intensPlottingInds[intensity], frequency))
                plt.plot(meanTraceEachSetting[intensity, frequency, :], 'k-')
                plt.ylim([minVoltageAllSettings, maxVoltageAllSettings])
                plt.axis('off')

        #This function returns the location of the text labels
        #We have to mess with the ideal locations due to the geometry of the plot
        def getXlabelpoints(n):
            rawArray = np.array(range(1, n+1))/float(n+1) #The positions in a perfect (0,1) world
            diffFromCenter = rawArray - 0.6
            partialDiffFromCenter = diffFromCenter * 0.175 #Percent change has to be determined empirically
            finalArray = rawArray - partialDiffFromCenter
            return finalArray

        #Not sure yet if similar modification to the locations will be necessary.
        def getYlabelpoints(n):
            rawArray = np.array(range(1, n+1))/float(n+1) #The positions in a perfect (0,1) world
            return rawArray

        freqLabelPositions = getXlabelpoints(len(possibleFreq))
        for indp, position in enumerate(freqLabelPositions):
            plt.figtext(position, 0.075, "%.1f"% (possibleFreq[indp]/1000), ha = 'center')

        intensLabelPositions = getYlabelpoints(len(possibleIntensity))
        for indp, position in enumerate(intensLabelPositions):
            plt.figtext(0.075, position, "%d"% possibleIntensity[indp])

        plt.figtext(0.525, 0.025, "Frequency (kHz)", ha = 'center')
        plt.figtext(0.025, 0.5, "Intensity (dB SPL)", va = 'center', rotation = 'vertical')
        plt.show()


    def flip_electrode_tuning(self, session, behavSuffix, electrodes=None , rasterRange=[-0.5, 1], tcRange=[0, 0.1]):

        if not electrodes:
            electrodes=self.defaultElectrodes

        plotTitle = self.loader.get_session_filename(session)

        spikesList=[]
        eventsList=[]
        freqList=[]
        rasterRangeList=[]
        tcRangeList=[]

        bdata = self.loader.get_session_behavior(behavSuffix)
        freqEachTrial = bdata['currentFreq']
        eventData = self.loader.get_session_events(session)
        eventOnsetTimes = self.loader.get_event_onset_times(eventData)

        for electrode in electrodes:
            spikeData = self.loader.get_session_spikes(session, electrode)
            spikeTimestamps = spikeData.timestamps

            spikesList.append(spikeTimestamps)
            eventsList.append(eventOnsetTimes)
            freqList.append(freqEachTrial)
            rasterRangeList.append(rasterRange)
            tcRangeList.append(tcRange)

        dataList=zip(spikesList, eventsList, freqList, electrodes, rasterRangeList, tcRangeList)

        self._electrode_tuning(dataList)



    @dataplotter.FlipThroughData
    def _electrode_tuning(spikeTimestamps, eventOnsetTimes, freqEachTrial, electrode, rasterRange, tcRange):

        '''
        Fix this so that
        '''

        #Unpack the data tuple (Watch out - make sure things from the above method are in the right order)
        # spikeTimestamps, eventOnsetTimes, freqEachTrial, electrode, rasterRange, tcRange = dataTuple

        possibleFreq=np.unique(freqEachTrial)
        freqLabels = ['{0:.1f}'.format(freq/1000.0) for freq in possibleFreq]
        fig = plt.gcf()

        ax1=plt.subplot2grid((3, 3), (0, 0), rowspan=3, colspan=2)
        dataplotter.plot_raster(spikeTimestamps, eventOnsetTimes, sortArray = freqEachTrial, ms=1, labels=freqLabels, timeRange=rasterRange)
        plt.title("Electrode {}".format(electrode))
        ax1.set_ylabel('Freq (kHz)')
        ax1.set_xlabel('Time from sound onset (sec)')

        ax2=plt.subplot2grid((3, 3), (0, 2), rowspan=3, colspan=1)
        dataplotter.one_axis_tc_or_rlf(spikeTimestamps, eventOnsetTimes, freqEachTrial, timeRange=tcRange)

        ax2.set_ylabel("Avg spikes in range {}".format(tcRange))
        ax2.set_xticks(range(len(freqLabels)))
        ax2.set_xticklabels(freqLabels, rotation='vertical')
        ax2.set_xlabel('Freq (kHz)')
