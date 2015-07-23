import os
import json
import itertools
from jaratoolbox.test.nick.ephysExperiments import ephys_experiment_v3 as ee3
# reload(ee3)


def convert_to_builtin_type(obj):
    d = {'__class__': obj.__class__.__name__,
         '__module__': obj.__module__,
         }
    d.update(obj.__dict__)
    return d


class RecordingDay(object):

    '''
    Parent class so that we don't have to re-enter this info for each recording site.
    The construction of new recording sites could possible be handled through a
    method of this class in the future. For now, an instance of this class is a
    required argument for the RecordingSite class
    '''

    def __init__(self, animalName, date, experimenter, **kwargs):
        self.animalName = animalName
        self.date = date
        self.experimenter = experimenter
        self.siteList = []
        # An internal instance of the ephys experiment class for easy plotting?
        self.ee = ee3.EphysExperiment(
            animalName,
            date,
            experimenter=experimenter,
            **kwargs)

    def add_site(self, depth, goodTetrodes):
        site = RecordingSite(depth,
                             goodTetrodes,
                             animalName=self.animalName,
                             date=self.date,
                             experimenter=self.experimenter)
        self.siteList.append(site)
        return site


class RecordingSite(object):

    '''
    Class for holding information about a recording site. Will act as a parent for all of the
    RecordingSession objects that hold information about each individual recording session.
    '''

    def __init__(self, depth, goodTetrodes, animalName, date, experimenter):

        self.animalName = animalName
        self.date = date
        self.experimenter = experimenter
        self.depth = depth
        self.goodTetrodes = goodTetrodes
        self.sessionList = []
        self.clusterList = []

    def add_session(self, sessionID, behavFileIdentifier, sessionType):
        session = RecordingSession(
            sessionID,
            behavFileIdentifier,
            sessionType,
            self.date)
        self.sessionList.append(session)
        return session

    def add_cluster(
            self,
            clusterNumber,
            tetrode,
            soundResponsive=False,
            laserPulseResponse=False,
            followsLaserTrain=False,
            comments=''):
        cluster = Cluster(
            self.animalName,
            self.date,
            self.depth,
            self.experimenter,
            cluster=clusterNumber,
            tetrode=tetrode,
            ephysSessionList=self.get_session_filenames(),
            behavFileList=self.get_session_behavIDs(),
            sessionTypes=self.get_session_types(),
            soundResponsive=soundResponsive,
            laserPulseResponse=laserPulseResponse,
            followsLaserTrain=followsLaserTrain,
            comments=comments)
        self.clusterList.append(cluster)
        return cluster

    def get_session_filenames(self):
        return [s.session for s in self.sessionList]

    def get_session_behavIDs(self):
        return [s.behavFileIdentifier for s in self.sessionList]

    def get_session_types(self):
        return [s.sessionType for s in self.sessionList]

    def get_session_inds_one_type(self, plotType, report):
        return [
            index for index,
            s in enumerate(
                self.sessionList) if (
                (s.plotType == plotType) & (
                    s.report == report))]

    def generate_main_report(self):
        '''
        Generate the reports for all of the sessions in this site. This is where we should interface with
        the multiunit clustering code, since all of the sessions that need to be clustered together have
        been defined at this point.

        FIXME: This method should be able to load some kind of report template perhaps, so that the
        types of reports we can make are not a limited. For instance, what happens when we just have
        rasters for a site and no tuning curve? Implementing this is a lower priority for now.

        Incorporated lan's code for plotting the cluster reports directly on the main report
        '''
        # FIXME: import another piece of code to do this?

        for tetrode in self.goodTetrodes:
            oneTT = cms2.MultipleSessionsToCluster(
                self.animalName,
                self.get_session_filenames(),
                tetrode,
                '{}at{}um'.format(
                    self.date,
                    self.depth))
            oneTT.load_all_waveforms()

            # Do the clustering if necessary.
            clusterFile = os.path.join(
                oneTT.clustersDir,
                'Tetrode%d.clu.1' %
                oneTT.tetrode)
            if os.path.isfile(clusterFile):
                oneTT.set_clusters_from_file()
            else:
                oneTT.create_multisession_fet_files()
                oneTT.run_clustering()
                oneTT.set_clusters_from_file()

            oneTT.save_single_session_clu_files()
            possibleClusters = np.unique(oneTT.clusters)

            exp2 = EphysExperiment(
                self.animalName,
                self.date,
                experimenter=self.experimenter)

            # Iterate through the clusters, making a new figure for each cluster.
            # for indClust, cluster in enumerate([3]):
            for indClust, cluster in enumerate(possibleClusters):

                mainRasterInds = self.get_session_inds_one_type(
                    plotType='raster',
                    report='main')
                mainRasterSessions = [
                    self.get_session_filenames()[i] for i in mainRasterInds]
                mainRasterTypes = [
                    self.get_session_types()[i] for i in mainRasterInds]

                mainTCinds = self.get_session_inds_one_type(
                    plotType='tc_heatmap',
                    report='main')
                mainTCsessions = [
                    self.get_session_filenames()[i] for i in mainTCinds]

                mainTCbehavIDs = [
                    self.get_session_behavIDs()[i] for i in mainTCinds]
                mainTCtypes = [self.get_session_types()[i] for i in mainTCinds]

                # The main report for this cluster/tetrode/session
                plt.figure()

                for indRaster, rasterSession in enumerate(mainRasterSessions):
                    plt.subplot2grid(
                        (6, 6), (indRaster, 0), rowspan=1, colspan=3)
                    exp2.plot_session_raster(
                        rasterSession,
                        tetrode,
                        cluster=cluster,
                        replace=1,
                        ms=1)
                    plt.ylabel(
                        '{}\n{}'.format(
                            mainRasterTypes[indRaster],
                            rasterSession.split('_')[1]),
                        fontsize=10)
                    ax = plt.gca()
                    extraplots.set_ticks_fontsize(ax, 6)

                # We can only do one main TC for now.
                plt.subplot2grid((6, 6), (0, 3), rowspan=3, colspan=3)
                #tcIndex = site.get_session_types().index('tuningCurve')
                tcSession = mainTCsessions[0]
                tcBehavID = mainTCbehavIDs[0]
                exp2.plot_session_tc_heatmap(
                    tcSession,
                    tetrode,
                    tcBehavID,
                    replace=1,
                    cluster=cluster)
                plt.title(
                    "{0}\nBehavFileID = '{1}'".format(
                        tcSession,
                        tcBehavID),
                    fontsize=10)

                nSpikes = len(oneTT.timestamps)
                nClusters = len(possibleClusters)
                #spikesEachCluster = np.empty((nClusters, nSpikes),dtype = bool)
                # if oneTT.clusters == None:
                # oneTT.set_clusters_from_file()
                # for indc, clusterID in enumerate (possibleClusters):
                #spikesEachCluster[indc, :] = (oneTT.clusters==clusterID)

                tsThisCluster = oneTT.timestamps[oneTT.clusters == cluster]
                wavesThisCluster = oneTT.samples[oneTT.clusters == cluster]
                # -- Plot ISI histogram --
                plt.subplot2grid((6, 6), (4, 0), rowspan=1, colspan=3)
                spikesorting.plot_isi_loghist(tsThisCluster)
                plt.ylabel(
                    'c%d' %
                    cluster,
                    rotation=0,
                    va='center',
                    ha='center')
                plt.xlabel('')

                # -- Plot waveforms --
                plt.subplot2grid((6, 6), (5, 0), rowspan=1, colspan=3)
                spikesorting.plot_waveforms(wavesThisCluster)

                # -- Plot projections --
                plt.subplot2grid((6, 6), (4, 3), rowspan=1, colspan=3)
                spikesorting.plot_projections(wavesThisCluster)

                # -- Plot events in time --
                plt.subplot2grid((6, 6), (5, 3), rowspan=1, colspan=3)
                spikesorting.plot_events_in_time(tsThisCluster)

                fig_path = oneTT.clustersDir
                fig_name = 'TT{0}Cluster{1}.png'.format(tetrode, cluster)
                full_fig_path = os.path.join(fig_path, fig_name)
                print full_fig_path
                # plt.tight_layout()
                plt.savefig(full_fig_path, format='png')
                # plt.show()
                plt.close()

            plt.figure()
            oneTT.save_multisession_report()
            plt.close()


class RecordingSession(object):

    '''
    Class to hold information about a single session.
    Includes the session name, the type of session, and any associated behavior data
    Accepts just the time of the recording (i.e. 11-36-54), and the date, which can
    be passed from the parent when this class is called. This keeps us from
    having to write it over and over again.
    '''

    def __init__(self, sessionID, behavFileIdentifier, sessionType, date):
        self.session = '_'.join([date, sessionID]),
        self.behavFileIdentifier = behavFileIdentifier,
        self.sessionType = sessionType,
        self.plotType = ''
        self.report = ''

    def set_plot_type(self, plotTypeStr, report='main'):
        self.plotType = plotTypeStr
        self.report = report


class Cluster(object):

    def __init__(
            self,
            animalName,
            date,
            depth,
            experimenter,
            clusterNumber,
            ephysSessionList,
            behavFileList,
            sessionTypes,
            soundResponsive,
            laserPulseResponse,
            followsLaserTrain,
            comments=''):

        self.animalName = animalName
        self.date = date
        self.depth = depth
        self.experimenter = experimenter
        self.cluster = clusterNumber
        self.tetrode = tetrode
        self.clusterNumber = clusterNumber
        self.ephysSessionList = ephysSessionList
        self.behavFileList = behavFileList
        self.sessionTypes = sessionTypes
        self.soundResponsive = soundResponsive
        self.laserPulseResponse = laserPulseResponse
        self.followsLaserTrain = followsLaserTrain
        self.comments = comments

        self.clusterID = '_'.join([animalName,
                                   date,
                                   str(depth),
                                   'TT{}'.format(self.tetrode),
                                   str(clusterNumber)])


class LoadedCluster(Cluster):

    def __init__(self, decodedDict):
        for key in decodedDict:
            if key not in [
                    '__class__',
                    '__module__']:  # FIXME: get rid of this, it does not seem useful
                setattr(self, key, decodedDict[key])


class JSONCellDB(object):

    def __init__(self, dbFilename):
        self.dbFilename = dbFilename
        clusterDB = self.connect(dbFilename)
        self.clusterList = self.build_clusters(clusterDB)
        self.clusterIDList = self.get_cluster_IDs(self.clusterList)

    def connect(self, dbFilename):
        if os.path.isfile(dbFilename):
            # This should open a new file if the file does not exist
            with open(dbFilename, 'r+') as f:
                clusterDB = json.load(f)
        else:  # No file to load - create it?
            clusterDB = []
        return clusterDB

    def build_clusters(self, clusterDB):
        loadedClusters = [LoadedCluster(d) for d in clusterDB]
        return loadedClusters

    def get_cluster_IDs(self, clusterList):
        clusterIDList = [
            c.clusterID.encode(
                'utf8',
                'ignore') for c in self.clusterList]
        return clusterIDList

    def add_clusters(self, clusterObjList):
        for clusterObj in clusterObjList:
            if clusterObj.clusterID not in self.clusterIDList:
                self.clusterList.append(clusterObj)
                self.clusterIDList = self.get_cluster_IDs(self.clusterList)
            else:
                print "Attempted to add duplicate cluster: {}".format(clusterObj.clusterID)

    def write_database(self):
        with open(self.dbFilename, 'w') as f:
            f.write('[\n')
            for indClust, clusterFile in enumerate(self.clusterList):
                f.write(
                    json.dumps(
                        clusterFile,
                        indent=4,
                        sort_keys=True,
                        default=convert_to_builtin_type))
                if indClust < len(self.clusterList) - 1:
                    f.write(',\n')
            f.write('\n]')
