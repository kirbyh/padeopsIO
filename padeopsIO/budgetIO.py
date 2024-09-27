import numpy as np
import os
import re
import warnings
import glob
from scipy.io import savemat, loadmat

import padeopsIO.budgetkey as budgetkey  # defines key pairing
import padeopsIO.inflow as inflow  # interface to retrieve inflow profiles
import padeopsIO.turbineArray as turbineArray  # reads in a turbine array similar to turbineMod.F90
from padeopsIO.io_utils import structure_to_dict, key_search_r
from padeopsIO.wake_utils import *
from padeopsIO.nml_utils import parser


class BudgetIO(): 
    """
    Class for reading and writing outputfiles to PadeOps. 
    """

    key = budgetkey.get_key()

    def __init__(self, dir_name, verbose=False, filename=None, 
                 runid=None, normalize_origin=False, 
                 padeops=False, npz=False, mat=False, npy=False, 
                 read_budgets=None, 
                ): 
        """
        Creates different instance variables depending on the keyword arguments given. 
        
        Every instance needs a directory name. If this object is reading information 
        from output files dumped by PadeOps, then this is the directory where those 
        files are stored. This object may also read information from a local subset of 
        saved data.  
        
        The BudgetIO class will try to initialize from source files if kwarg 
        `padeops=True` is given. Alternatively, initialize from .mat files using kwarg
        `mat=True`. 
        
        If those keyword arguments not are present, then the directory name will (attempt to) read from 
        budgets of saved .npz files. 

        Regardless of the method of reading budget files, __init__ will initialize the following fields: 
        RUN INFORMATION: 
            filename, dir_name, 
        DOMAIN VARIABLES: 
            Lx, Ly, Lz, nx, ny, nz, dx, dy, dz, xLine, yLine, zLine, 
        TURBINE VARIABLES: 
            nTurb, 
        PHYSICS: 
            Re, Ro, 
        BUDGET VARIABLES: 
            last_tidx, last_n, 
        
        """

        # print statements? default False
        if verbose: 
            self.verbose = True 
            print('Attempting to initialize BudgetIO object at', dir_name)
        else: 
            self.verbose = False
        
        self.dir_name = dir_name
        
        # all files associated with this case will begin with <filename>
        if filename is None: 
            # defaults to the directory name, split on non-word characters
            dir_list = re.split('\W+', dir_name)
            # pick the last non-empty string
            self.filename = next(s for s in reversed(dir_list) if s)
        else: 
            self.filename = filename

        self.filename_budgets = self.filename + '_budgets'  # standardize this
        
        # ========== Associate files ==========

        # if we are given the required keywords, try to initialize from PadeOps source files
        self.associate_padeops = False
        self.associate_npz = False
        self.associate_mat = False
        self.associate_nml = False
        self.associate_fields = False 
        self.associate_budgets = False
        self.associate_grid = False
        self.associate_turbines = False
        self.normalized_xyz = False

        if padeops: 
            try: 
                self._init_padeops(runid=runid, normalize_origin=normalize_origin)
                # self.associate_padeops = True  # moved this inside _init_padeops()
                if self.verbose: 
                    print('Initialized BudgetIO at ' + dir_name + ' from PadeOps source files.')

            except OSError as err: 
                print('Attempted to read PadeOps output files, but at least one was missing.')
                print(err)
                raise
        
        elif mat:  # .mat save files
            self._init_mat()

            if self.verbose: 
                print('Initialized BudgetIO at ' + dir_name + ' from .mat files. ')

        elif npz:  # .npz save files
            self._init_npz()

            if self.verbose: 
                print('Initialized BudgetIO at ' + dir_name + ' from .npz files. ')
        
        elif npy: 
            self._init_npy(normalize_origin=normalize_origin)
            if self.verbose: 
                print('Initialized BudgetIO at ' + dir_name + ' from .npy files. ')
        
        else: 
            raise AttributeError("__init__(): ")

        self.budget = {}  # empty dictionary

        if read_budgets is not None: 
            # if read_budgets passed in as keyword argument, read budgets on initialization
            self.read_budgets(budget_terms=read_budgets)
    

    def _init_padeops(self, runid=None, normalize_origin=False): 
        """
        Initializes source files to be read from output files in PadeOps. 
        
        Raises OSError if source files cannot be read
        """

        # parse namelist 
        try: 
            self._read_inputfile(runid=runid)  # this initializes convenience variables
            
        except IndexError as err: 
            warnings.warn("_init_padeops(): Could not find input file. Perhaps the directory does not exist? ")
            print(err)
            raise err
        
        # default time ID for initialization
        self.tidx = 0  # tidx defualts to zero
        self.time = 0

        # READ TURBINES, only do this if usewindturbines = True
        if self.associate_nml and self.input_nml['windturbines']['usewindturbines']:  # TODO may throw KeyError 
            
            if self.verbose: 
                print('_init_padeops(): Initializing wind turbine array object')
                
            turb_dir = self.input_nml['windturbines']['turbinfodir']
            if not os.path.exists(turb_dir):  # switch this to pathlib
                # hotfix: maybe this folder was copied elsewhere
                turb_dir = os.path.join(self.dir_name, 'turb')
                
            num_turbines = self.input_nml['windturbines']['num_turbines']
            ADM_type = self.input_nml['windturbines']['adm_type']
            try: 
                self.turbineArray = turbineArray.TurbineArray(turb_dir, 
                                                              num_turbines=num_turbines, 
                                                              ADM_type=ADM_type, 
                                                              verbose=self.verbose)
                self.associate_turbines = True

                if self.verbose: 
                    print('_init_padeops(): Finished initializing wind turbine array with {:d} turbine(s)'.format(num_turbines))

            except FileNotFoundError as e: 
                warnings.warn("Turbine file not found, bypassing associating turbines.")
                self.turbineArray = None
                if self.verbose: 
                    print(e)

        # Throw an error if no RunID is found 
        if 'runid' not in self.__dict__:
            raise AttributeError("No RunID found. To explicitly pass one in, use kwarg: runid=")
                    
        # loads the grid, normalizes if `associate_turbines=True` and `normalize_origin='turb'`
        # (Should be done AFTER loading turbines to normalize origin)
        if not self.associate_grid: 
            self._load_grid(normalize_origin=normalize_origin)
            
        # object is reading from PadeOps output files directly
        if self.verbose: 
            print('BudgetIO initialized using info files at time:' + '{:.06f}'.format(self.time))
            
        # at this point, we are reading from PadeOps output files
        self.associate_padeops = True
        
        # try to associate fields
        self.field = {}
        try: 
            self.last_tidx = self.unique_tidx(return_last=True)  # last tidx in the run with fields
            self.associate_fields = True

        except FileNotFoundError as e: 
            warnings.warn("_init_padeops(): No field files found!")
            if self.verbose: 
                print("\tNo field files found!")
        
        # try to associate budgets
        try: 
            self.all_budget_tidx = self.unique_budget_tidx(return_last=False)
            self.associate_budgets = True
        except FileNotFoundError as e: 
            warnings.warn("_init_padeops(): No budget files found!")
            if self.verbose: 
                print("\tNo budget files found.")
            
        if self.associate_fields: # The following are initialized as the final saved instanteous field and budget: 
            self.field_tidx = self.last_tidx

        if self.associate_budgets: 
            self.last_n = self.last_budget_n()  # last tidx with an associated budget
            self.budget_tidx = self.unique_budget_tidx()  # but may be changed by the user
            self.budget_n = self.last_n
            
    
    def _read_inputfile(self, runid=None): 
        """
        Reads the input file (Fortran 90 namelist) associated with the CFD simulation. 

        Parameters
        ----------
        runid : int
            RunID number to try and match, given inputfiles self.dir_name. 
            Default: None
        
        Returns
        -------
        None
        """
                        
        # search all files ending in '*.dat' 
        inputfile_ls = glob.glob(self.dir_name + os.sep + '*.dat')  # for now, just search this 
        
        if len(inputfile_ls) == 0: 
            raise FileNotFoundError('_read_inputfile(): No inputfiles found at {:s}'.format(self.dir_name))
            
        if self.verbose: 
            print("\tFound the following files:", inputfile_ls)

        # try to search all input files '*.dat' for the proper run and match it
        for inputfile in glob.glob(self.dir_name + os.sep + '*.dat'): 
            input_nml = parser(inputfile)
            if self.verbose: 
                print('\t_read_inputfile(): trying inputfile', inputfile)

            try: 
                tmp_runid = input_nml['io']['runid']
            except KeyError as e: 
                if self.verbose: 
                    print('\t_read_inputfile(): no runid for', inputfile)
                tmp_runid = None  # not all input files have a RunID
        
            if runid is not None: 
                if tmp_runid == runid: 
                    self.input_nml = input_nml
                    self._convenience_variables()  # make some variables in the metadata more accessible, also loads grid
                    self.associate_nml = True  # successfully loaded input file

                    if self.verbose: 
                        print('\t_read_inputfile(): matched RunID with', inputfile)
                    return
            elif self.verbose: 
                print("\t_read_inputfile(): WARNING - no keyword `runid` given to init.")

        # if there are still no input files found, we've got a problem 
        
        warnings.warn('_read_inputfile(): No match to given `runid`, picking the first inputfile to read.')
        
        if self.verbose: 
            print("\t_read_inputfile(): Reading namelist file from {}".format(inputfile_ls[0]))
            
        self.input_nml = parser(inputfile_ls[0])
        self._convenience_variables()  # make some variables in the metadata more accessible
        self.associate_nml = True  # successfully loaded input file

        
    def _convenience_variables(self): 
        """
        Aside from reading in the Namelist, which has all of the metadata, also make some
        parameters more accessible. 
        
        Called by _read_inputfile() and by _init_npz()
        
        Special note: these are all lower case when reading from the dictionary or namelist! 
        """
                
        # RUN VARIABLES: 
        self.runid = self.input_nml['io']['runid']
                
        # TURBINE VARIABLES: 
        self.nTurb = self.input_nml['windturbines']['num_turbines']

        # PHYSICS: 
        if self.input_nml['physics']['isinviscid']:  # boolean
            self.Re = np.Inf
        else: 
            self.Re = self.input_nml['physics']['re']

        if self.input_nml['physics']['usecoriolis']: 
            self.Ro = key_search_r(self.input_nml, 'ro') 
            self.lat = key_search_r(self.input_nml, 'latitude')
            self.Ro_f = self.Ro / (2*np.cos(self.lat*np.pi/180))
        else: 
            self.Ro = np.Inf

        if key_search_r(self.input_nml, 'isstratified'): 
            self.Fr = key_search_r(self.input_nml, 'fr')
        else: 
            self.Fr = np.Inf
        
        self.galpha = key_search_r(self.input_nml, 'g_alpha')

    
    def _load_grid(self, x=None, y=None, z=None, origin=(0, 0, 0), normalize_origin=None): 
        """
        Creates dx, dy, dz, and xLine, yLine, zLine variables. 
        
        Expects (self.)Lx, Ly, Lz, nx, ny, nz in kwargs or in self.input_nml
        """

        if self.associate_grid and self.verbose: 
            print("_load_grid(): Grid already exists. ")
            return
        
        # build domain: 
        if x is not None and y is not None and z is not None: 
            for xi, xname in zip([x, y, z], ['x', 'y', 'z']): 
                xi = np.atleast_1d(xi)  # expand to at least 1D
                self.__dict__[f'{xname}Line'] = xi
                self.__dict__[f'L{xname}'] = xi.max() - xi.min()
                try: 
                    self.__dict__[f'd{xname}'] = xi[1]-xi[0]
                    self.__dict__[f'n{xname}'] = len(xi)
                except IndexError:  # 1D along this axis
                    self.__dict__[f'd{xname}'] = None  # does not make sense to have dxi 
                    self.__dict__[f'n{xname}'] = 1
        else: 
            terms_map = {'nx': 'nx', 'ny': 'ny', 'nz': 'nz', 
                        'lx': 'Lx', 'ly': 'Ly', 'lz': 'Lz'}  # search -> keys, name -> values
            
            for key in terms_map: 
                    self.__dict__[terms_map[key]] = key_search_r(self.input_nml, key)

            self.dx = self.Lx/self.nx
            self.dy = self.Ly/self.ny
            self.dz = self.Lz/self.nz

            self.xLine = np.linspace(0,self.Lx-self.dx,self.nx)
            self.yLine = np.linspace(0,self.Ly-self.dy,self.ny)
            # staggered in z
            self.zLine = np.linspace(self.dz/2,self.Lz-(self.dz/2),self.nz)
        
        self.origin = origin  # default origin location

        self.associate_grid = True

        if normalize_origin:  # not None or False
            if str(normalize_origin) in ['turb', 'turbine'] and self.associate_turbines: 
                self.turbineArray.set_sort('xloc', sort=True)
                self.normalize_origin(self.turbineArray.turbines[0].pos)
            
            else: 
                self.normalize_origin(normalize_origin)  # expects tuple (x, y, z)


    def normalize_origin(self, xyz): 
        """
        Normalize the origin to point xyz
        
        Parameters
        ----------
        xyz : None or tuple
            If tuple, moves the origin to (x, y, z)
            If none, resets the origin. 
        """
        
        if xyz is not None: 
            self.xLine -= (xyz[0] - self.origin[0])
            self.yLine -= (xyz[1] - self.origin[1])
            self.zLine -= (xyz[2] - self.origin[2])
            self.normalized_xyz=True
            self.origin = xyz

        else: 
            self.xLine += self.origin[0]
            self.yLine += self.origin[1]
            self.zLine += self.origin[2]
            self.normalized_xyz=False
            self.origin = (0, 0, 0)


    def _init_npz(self, normalize_origin=False): 
        """
        Initializes the BudgetIO object by attempting to read .npz files saved from a previous BudgetIO object 
        from write_npz(). 

        Expects target files: 
        One filename including "{filename}_budgets.npz"
        One filename including "_metadata.npz"
        """
        # load metadata: expects a file named <filename>_metadata.npz
        filepath = self.dir_name + os.sep + self.filename + '_metadata.npz'
        try: 
            ret = np.load(filepath, allow_pickle=True)
        except FileNotFoundError as e: 
            raise e
        
        self.input_nml = ret['input_nml'].item()
        self.associate_nml = True

        if 'turbineArray' in ret.files: 
            init_dict = ret['turbineArray'].item()
            init_ls = [t for t in init_dict['turbines']]
            self.turbineArray = turbineArray.TurbineArray(init_ls=init_ls)
            self.associate_turbines = True

        origin = (0, 0, 0)
        if 'origin' in ret.files: 
            origin = ret['origin']

        # set convenience variables: 
        self._convenience_variables()
        self.associate_nml = True
        
        if not self.associate_grid: 
            self._load_grid(x=np.squeeze(ret['x']), 
                            y=np.squeeze(ret['y']), 
                            z=np.squeeze(ret['z']), 
                            origin=origin, 
                            normalize_origin=normalize_origin)

       # check budget files
        budget_files = glob.glob(self.dir_name + os.sep + self.filename_budgets + '.npz')
        if len(budget_files) == 0: 
            warnings.warn("No associated budget files found")
        else: 
            self.associate_budgets = True
            self.budget_n = None
            self.budget_tidx = None  
            self.last_n = None  # all these are missing in npz files 04/24/2023

        if self.verbose: 
            print('_init_npz(): BudgetIO initialized using .npz files.')

        self.associate_npz = True


    def _init_npy(self, **kwargs): 
        """
        WARNING: Deprecated feature, use _init_npz() instead. 

        Initializes the BudgetIO object by attempting to read .npy metadata files
        saved from a previous BudgetIO object from write_npz(). 

        Expects target files: 
        One filename including "{filename}_budgets.npz"
        One filename including "_metadata.npy"
        """
        print('_init_npy(): Warning - deprecated. Use _init_npz() instead. ')
        
         # load metadata: expects a file named <filename>_metadata.npy
        filepath = self.dir_name + os.sep + self.filename + '_metadata.npy'
        try: 
            self.input_nml = np.load(filepath, allow_pickle=True).item()
        except FileNotFoundError as e: 
            raise e
        
       # check budget files
        budget_files = glob.glob(self.dir_name + os.sep + self.filename_budgets + '.npz')
        if len(budget_files) == 0: 
            warnings.warn("No associated budget files found")
        else: 
            self.associate_budgets = True
            self.budget_n = None
            self.budget_tidx = None  
            self.last_n = None  # all these are missing in npy files 04/24/2023
        
        # attempt to load turbine file - need this before loading grid
        if 'auxiliary' in self.input_nml.keys() and 'turbineArray' in self.input_nml['auxiliary']: 
            self.turbineArray = turbineArray.TurbineArray(
                init_dict=self.input_nml['auxiliary']['turbineArray']
                )
            self.associate_turbines = True
        
        self._convenience_variables()
        self.associate_nml = True
        self.associate_npz = True
        
        if not self.associate_grid: 
            self._load_grid(**kwargs)

        if self.verbose: 
            print('_init_npz(): BudgetIO initialized using .npz files.')


    def _init_mat(self, normalize_origin=False): 
        """
        Initializes the BudgetIO object by attempting to read .npz files saved from a previous
        BudgetIO object from write_mat(). 

        Expects target files: 
        One filename including "{filename}_budgets.mat"
        One filename including "{filename}_metadata.mat"
        """

        # load metadata: expects a file named <filename>_metadata.mat
        filepath = self.dir_name + os.sep + self.filename + '_metadata.mat'
        try: 
            ret = loadmat(filepath)
        except FileNotFoundError as e: 
            raise e
        
        self.input_nml = structure_to_dict(ret['input_nml'])
        self.associate_nml = True

        if 'turbineArray' in ret.keys(): 
            init_dict = structure_to_dict(ret['turbineArray'])
            init_ls = [structure_to_dict(t) for t in init_dict['turbines']]
            self.turbineArray = turbineArray.TurbineArray(init_ls=init_ls)
            self.associate_turbines = True

        origin = (0, 0, 0)
        if 'origin' in ret.keys(): 
            origin = ret['origin']

        # set convenience variables: 
        self._convenience_variables()
        self.associate_nml = True
        
        if not self.associate_grid: 
            self._load_grid(x=np.squeeze(ret['x']), 
                            y=np.squeeze(ret['y']), 
                            z=np.squeeze(ret['z']), 
                            origin=origin, 
                            normalize_origin=normalize_origin)

        # link budgets
        budget_files = glob.glob(self.dir_name + os.sep + self.filename_budgets + '.mat')
        if len(budget_files) == 0: 
            warnings.warn("No associated budget files found")
        else: 
            self.associate_budgets = True
            self.budget_n = None
            self.budget_tidx = None  
            self.last_n = None  # all these are missing in .mat files 07/03/2023

        if self.verbose: 
            print('_init_mat(): BudgetIO initialized using .mat files.')

        self.associate_mat = True


    def set_filename(self, filename): 
        """
        Changes the filename associated with this object. 

        Make sure filename_budgets is consistent with __init__()
        """
        self.filename = filename
        self.filename_budgets = filename + "_budgets"
        
        
    def write_npz(self, write_dir=None, budget_terms='default', filename=None, overwrite=False, 
                  xlim=None, ylim=None, zlim=None): 
        """
        Saves budgets as .npz files. Each budget receives its own .npz file, with the fourth dimension representing
        the budget term number (minus one, because python indexing starts at zero). 
        
        Budgets are defined in e.g. PadeOps/src/incompressible/budget_time_avg.F90. See budget_key.py
        From a high level: 
            Budget 0: Mean quantities (1st and 2nd order)
            Budget 1: Momentum budget terms
            Budget 2: MKE budget terms
            Budget 3: TKE budget terms
            
        parameters 
        ----------
        write_dir : str
            Location to write .npz files. Default: same directory as self.outputdir_name
        budget_terms : list 
            Budget terms to be saved (see ._parse_budget_terms()). Alternatively, 
            use 'current' to save the budget terms that are currently loaded. 
        filename : str
            Sets the filename of written files
        overwrite : bool
            If true, will overwrite existing .npz files. 
        xlim, ylim, zlim : slice bounds
            See BudgetIO.slice()  # TODO: SAVE X,Y,Z information of slices
        """
        
        if not self.associate_budgets: 
            warnings.warn('write_npz(): No budgets associated! ') 
            return 
        
        # declare directory to write to, default to the working directory
        if write_dir is None: 
            write_dir = self.dir_name
        
        if budget_terms=='current': 
            # these are the currently loaded budgets
            key_subset = self.budget.keys()
            
        else: 
            # need to parse budget_terms with the key
            key_subset = self._parse_budget_terms(budget_terms)

        # load budgets
        sl = self.slice(budget_terms=key_subset, xlim=xlim, ylim=ylim, zlim=zlim)

        # if `filename` is provided, write files with the provided name
        if filename is None: 
            filename = self.filename

        filepath = write_dir + os.sep + filename + '_budgets.npz'
        
        # don't unintentionally overwrite files... 
        write_arrs = False 
        if not os.path.exists(filepath): 
            write_arrs = True

        elif overwrite: 
            warnings.warn("File already exists, overwriting... ")
            write_arrs = True

        else: 
            warnings.warn("Existing files found. Failed to write; try passing overwrite=True to override.")
            return

        save_arrs = {}
        for key in key_subset: 
            # crop the domain of the budgets here: 
            save_arrs[key] = sl[key]

        # write npz files! 
        if write_arrs: 
            np.savez(filepath, **save_arrs)
            
            # SAVE METADATA
            self.write_metadata(write_dir, filename, 'npz', sl['x'], sl['y'], sl['z']) 
            
            if self.verbose: 
                print("write_npz: Successfully saved the following budgets: ", list(key_subset))
                print("at " + filepath)
        
        
    def write_metadata(self, write_dir, fname, src, x, y, z): 
        """
        The saved budgets aren't useful on their own unless we also save some information like the mesh
        used in the simulation and some other information like the physical setup. That goes here. 
        
        """
        
        save_vars = ['input_nml'] #, 'xLine', 'yLine', 'zLine']
        save_dict = {key: self.__dict__[key] for key in save_vars}
        save_dict['x'] = x
        save_dict['y'] = y
        save_dict['z'] = z
        save_dict['origin'] = self.origin
        
        if self.associate_turbines: 
            save_dict['turbineArray'] = self.turbineArray.todict()
            for k in range(self.turbineArray.num_turbines): 
                # write turbine information
                for prop in ['power', 'uvel', 'vvel']: 
                    try: 
                        save_dict['t{:d}_{:s}'.format(k+1, prop)] = self.read_turb_property(tidx='all', prop_str=prop, steady=False, turb=k+1)
                    except KeyError as e: 
                        pass
            
        filepath_meta = os.path.join(write_dir, fname + f'_metadata.{src}')

        if src == 'mat': 
            savemat(filepath_meta, save_dict)
        
        elif src == 'npz': 
            np.savez(filepath_meta, **save_dict)
        
        if self.verbose: 
            print(f'write_metadata(): metadata written to {filepath_meta}')
            
            
    def write_mat(self, write_dir=None, budget_terms='default', 
                  filename=None, overwrite=False, 
                  xlim=None, ylim=None, zlim=None): 
        """
        Saves budgets as .mat (MATLAB) files. This is lazy code copied from write_npz(). 
            
        Parameters 
        ----------
        write_dir : str
            location to write .mat files. Default: same directory as self.outputdir_name
        budget_terms : list
            budget terms to be saved (see ._parse_budget_terms()). Alternatively, 
            use 'current' to save the budget terms that are currently loaded. 
        filename : str
            Sets the filename of written files
        overwrite : bool
            if true, will overwrite existing .mat files. 
        xlim, ylim, zlim : slice bounds
            see BudgetIO.slice()
        """
        
        if not self.associate_budgets: 
            warnings.warn('write_mat(): No budgets associated! ') 
            return 
        
        # declare directory to write to, default to the working directory
        if write_dir is None: 
            write_dir = self.dir_name
        
        # load budgets
        key_subset = self._parse_budget_terms(budget_terms)

        # load budgets
        sl = self.slice(budget_terms=key_subset, xlim=xlim, ylim=ylim, zlim=zlim)

        if filename is None: 
            filename = self.filename
        
        filepath = write_dir + os.sep + filename + '_budgets.mat'
        
        # don't unintentionally overwrite files... 
        write_arrs = False  
        if not os.path.exists(filepath): 
            write_arrs = True

        elif overwrite: 
            warnings.warn("File already exists, overwriting... ")
            write_arrs = True

        else: 
            warnings.warn("Existing files found. Failed to write; try passing overwrite=True to override.")
            return

        save_arrs = {}
        for key in key_subset: 
            save_arrs[key] = sl[key]

        # write mat files! 
        if write_arrs: 
            if self.verbose: 
                print('write_mat(): attempting to save budgets to', filepath)
                
            savemat(filepath, save_arrs)
            
            # SAVE METADATA HERE: 
            self.write_metadata(write_dir, filename, 'mat', sl['x'], sl['y'], sl['z'])
            
            if self.verbose: 
                print("write_mat(): Successfully saved the following budgets: ", list(key_subset))
                print("at" + filepath)

            
    def read_fields(self, field_terms=None, tidx=None): 
        """
        Reads fields from PadeOps output files into the self.field dictionary. 
        
        Parameters
        ----------
        field_terms : list
            list of field terms to read, must be be limited to: 
            'u', 'v', 'w', 'p', 'T'
        tidx : int, optional
            reads fields from the specified time ID. Default: self.last_tidx 
        
        Returns
        -------
        None
            Read fields are saved in self.field
        """
        
        if not self.associate_fields: 
           raise AttributeError("read_fields(): No fields linked. ")
        
        dict_match = {
            'u':'uVel', 
            'v':'vVel', 
            'w':'wVel', 
            'p':'prss', 
            'T':'potT', 
            'pfrn': 'pfrn',  # fringe pressure
            'pdns': 'pdns',  # DNS pressure... what is this? 
            'ptrb': 'ptrb',  # turbine pressure... what is this? 
        }  # add more?
        
        # parse terms: 
        if field_terms is None: 
            terms = dict_match.keys()
            
        else: 
            terms = [t for t in field_terms if t in dict_match.keys()]
        
        # parse tidx
        if tidx is None: 
            tidx = self.last_tidx
            
        # update self.time and self.tidx: 
        self.tidx = tidx
        
        info_fname = self.dir_name + '/Run{:02d}_info_t{:06d}.out'.format(self.runid, self.tidx)
        self.info = np.genfromtxt(info_fname, dtype=None)
        self.time = self.info[0]
        
        # the following is very similar to PadeOpsViz.ReadVelocities()
        
        for term in terms:             
            fname = self.dir_name + '/Run{:02d}_{:s}_t{:06d}.out'.format(self.runid, dict_match[term], tidx)
            tmp = np.fromfile(fname, dtype=np.dtype(np.float64), count=-1)
            self.field[term] = tmp.reshape((self.nx,self.ny,self.nz), order='F')  # reshape into a 3D array
                    
        print('BudgetIO loaded fields {:s} at time: {:.06f}'.format(str(list(terms)), self.time))
        
        
    def clear_budgets(self): 
        """
        Clears any loaded budgets. 

        Returns
        -------
        keys (list) : list of cleared budgets. 
        """
        if not self.associate_budgets: 
            if self.verbose: 
                print('clear_budgets(): no budgets to clear. ')
            return
        
        loaded_keys = self.budget.keys()
        self.budget = {}  # empty dictionary
        self.budget_n = None
        self.budget_tidx = None  #self.unique_budget_tidx(return_last=True)  # reset to final TIDX

        if self.verbose: 
            print('clear_budgets(): Cleared loaded budgets: {}'.format(loaded_keys))
        
        return loaded_keys

    
    def read_budgets(self, budget_terms='default', mmap=None, overwrite=False, tidx=None): 
        """
        Accompanying method to write_budgets. Reads budgets saved as .npz files 
        
        Parameters 
        ----------
        budget_terms : list 
            Budget terms (see ._parse_budget_terms() and budgetkey.py)
        mmap : str, optional
            Default None. Sets the memory-map settings in numpy.load(). 
            Expects None, 'r+', 'r', 'w+', 'c'
        overwrite : bool, optional
            If True, re-loads budgets that have already been loaded. Default False; 
            checks existing budgets before loading new ones. 
        tidx : int, optional
            If given, requests budget dumps at a specific time ID. Default None. This only affects
            reading from PadeOps output files; .npz and .mat are limited to one saved tidx. 

        Returns
        -------
        None
            Saves result in self.budget
        """
        
        if not self.associate_budgets: 
           raise AttributeError("read_budgets(): No budgets linked. ")
        
        # we need to handle computed quantities differently... 
        if any(t in ['uwake', 'vwake', 'wwake'] for t in budget_terms): 
            self.calc_wake()
            
            if self.verbose: 
                print("read_budgets: Successfully loaded wake budgets. ")


        # parse budget_terms with the key
        key_subset = self._parse_budget_terms(budget_terms, include_wakes=False)
        
        if self.budget_tidx == tidx:  # note: tidx could be `None`
            if not overwrite:  
                remove_keys = [key for key in key_subset if key in self.budget.keys()]
                if len(remove_keys) > 0 and self.verbose: 
                    print("read_budgets(): requested budgets that have already been loaded. \
                        \n  Removed the following: {}. Pass overwrite=True to read budgets anyway.".format(remove_keys))

                # remove items that have already been loaded in  
                key_subset = {key:key_subset[key] for key in key_subset if key not in self.budget.keys()}
                
            else: 
                self.clear_budgets()
                
        elif self.budget.keys() is not None and tidx is not None:  # clear previous TIDX budgets, if they exist
            self.clear_budgets()

        if self.associate_padeops: 
            self._read_budgets_padeops(key_subset, tidx=tidx)  # this will not include wake budgets
        elif self.associate_npz: 
            self._read_budgets_npz(key_subset, mmap=mmap)
        elif self.associate_mat: 
            self._read_budgets_mat(key_subset)
        else: 
            raise AttributeError('read_budgets(): No budgets linked. ')
        
        if self.verbose and len(key_subset) > 0: 
            print("read_budgets: Successfully loaded budgets. ")
        

    def _read_budgets_padeops(self, key_subset, tidx): 
        """
        Uses a method similar to ReadVelocities_Budget() in PadeOpsViz to read and store full-field budget terms. 
        """
        
        if tidx is None: 
            if self.budget or  self.budget_tidx is not None:  # empty dictionary evaluates to False in python
                # if there are budgets loaded, continue loading from that TIDX
                tidx = self.budget_tidx  
            else:  
                # load budgets from the last available TIDX
                tidx = self.unique_budget_tidx(return_last=True)
            
        elif tidx not in self.all_budget_tidx: 
            # find the nearest that actually exists
            tidx_arr = np.array(self.all_budget_tidx)
            closest_tidx = tidx_arr[np.argmin(np.abs(tidx_arr-tidx))]
            
            print("Requested budget tidx={:d} could not be found. Using tidx={:d} instead.".format(tidx, closest_tidx))
            tidx = closest_tidx 
            
        if self.verbose: 
            print(f'Loading budgets {list(key_subset.keys())} from {tidx}')
            
        # these lines are almost verbatim from PadeOpsViz.py
        for key in key_subset:
            budget, term = BudgetIO.key[key]
            
            searchstr =  self.dir_name + '/Run{:02d}_budget{:01d}_term{:02d}_t{:06d}_*.s3D'.format(self.runid, budget, term, tidx)
            u_fname = glob.glob(searchstr)[0]  
            
            self.budget_n = int(re.findall('.*_t\d+_n(\d+)', u_fname)[0])  # extract n from string
            self.budget_tidx = tidx  # update self.budget_tidx
            
            temp = np.fromfile(u_fname, dtype=np.dtype(np.float64), count=-1)
            self.budget[key] = temp.reshape((self.nx,self.ny,self.nz), order='F')  # reshape into a 3D array

        if self.verbose and len(key_subset) > 0: 
            print('BudgetIO loaded the budget fields at TIDX:' + '{:.06f}'.format(tidx))


    def _read_budgets_padeops_2d(self, key_subset, tidx):
        """
        Uses a method similar to ReadVelocities_Budget() in PadeOpsViz to read and store full-field budget terms.
        """
        if tidx is None:
            if self.budget or  self.budget_tidx is not None:  # empty dictionary evaluates to False in python
                # if there are budgets loaded, continue loading from that TIDX
                tidx = self.budget_tidx
            else:
                # load budgets from the last available TIDX
                tidx = self.unique_budget_tidx(return_last=True)

        elif tidx not in self.all_budget_tidx:
            # find the nearest that actually exists
            tidx_arr = np.array(self.all_budget_tidx)
            closest_tidx = tidx_arr[np.argmin(np.abs(tidx_arr-tidx))]

            print("Requested budget tidx={:d} could not be found. Using tidx={:d} instead.".format(tidx, closest_tidx))
            tidx = closest_tidx

        if self.verbose:
            print(f'Loading budgets {list(key_subset.keys())} from {tidx}')

        # these lines are almost verbatim from PadeOpsViz.py
        for key in key_subset:
            budget, term = BudgetIO.key[key]

            searchstr =  self.dir_name + '/Run{:02d}_budget{:01d}_term{:02d}_t{:06d}_*.s2D'.format(self.runid, budget, term, tidx)
            u_fname = glob.glob(searchstr)[0]

            self.budget_n = int(re.findall('.*_t\d+_n(\d+)', u_fname)[0])  # extract n from string
            self.budget_tidx = tidx  # update self.budget_tidx

            temp = np.fromfile(u_fname, dtype=np.dtype(np.float64), count=-1)
            self.budget[key] = temp.reshape((self.ny,self.nz), order='F')  # reshape into a 3D array

        if self.verbose and len(key_subset) > 0:
            print('BudgetIO loaded the budget fields at TIDX:' + '{:.06f}'.format(tidx))


    def _read_budgets_npz(self, key_subset, mmap=None): 
        """
        Reads budgets written by .write_npz() and loads them into memory
        """

        # load the npz file and keep the requested budget keys
        for key in key_subset: 
            npz = np.load(self.dir_name + os.sep + self.filename + '_budgets.npz')
            self.budget[key] = npz[key]  

        if self.verbose: 
            print('BudgetIO loaded the following budgets from .npz: ', list(key_subset.keys()))


    def _read_budgets_mat(self, key_subset): 
        """
        Reads budgets written by .write_mat()
        """

        for key in key_subset: 
            budgets = loadmat(self.dir_name + os.sep + self.filename + '_budgets.mat')
            self.budget[key] = budgets[key]  

        if self.verbose: 
            print('BudgetIO loaded the following budgets from .mat: ', list(key_subset.keys()))


    def _parse_budget_terms(self, budget_terms, include_wakes=False): 
        """
        Takes a list of budget terms, either keyed in index form (budget #, term #) or in common form (e.g. ['u_bar', 'v_bar'])
        and returns a subset of the `keys` dictionary that matches two together. `keys` dictionary is always keyed in common form. 

        budget_terms can also be a string: 'all', or 'default'. 

        'default' tries to load the following: 
            Budget 0 terms: ubar, vbar, wbar, all Reynolds stresses, and p_bar
            Budget 1 terms: all momentum terms
        'all' checks what budgets exist and tries to load them all. 

        For more information on the bi-directional keys, see budget_key.py
        
        Arguments
        ---------
        budget_terms : list of strings or string, see above
        include_wakes : bool, optional
            includes wake budgets if True, default False. 
        """

        # add string shortcuts here... # TODO move shortcuts to budgetkey.py? 
        if budget_terms=='default': 
            budget_terms = ['ubar', 'vbar', 'wbar', 
                            'tau11', 'tau12', 'tau13', 'tau22', 'tau23', 'tau33', 
                            'pbar']
            
        elif budget_terms=='current': 
            budget_terms = list(self.budget.keys())

        elif budget_terms=='all': 
            budget_terms = self.existing_terms(include_wakes=include_wakes)
            
        elif budget_terms=='RANS': 
            budget_terms = ['ubar', 'vbar', 'wbar', 
                            'pbar', 'Tbar', 
                            'uu', 'uv', 'uw', 'vv', 'vw', 'ww', 
                            'dpdx', 'dpdy', 'dpdz',
                            'tau11', 'tau12', 'tau13', 'tau22', 'tau23', 'tau33']

        elif type(budget_terms)==str: 
            warnings.warn("keyword argument budget_terms must be either 'default', 'all', 'RANS' or a list.")
            return {}  # empty dictionary

        # parse through terms: they are either 1) valid, 2) missing (but valid keys), or 3) invalid (not in BudgetIO.key)

        existing_keys = self.existing_terms(include_wakes=include_wakes)
        existing_tup = [BudgetIO.key[key] for key in existing_keys]  # corresponding associated tuples (#, #)

        valid_keys = [t for t in budget_terms if t in existing_keys]
        missing_keys = [t for t in budget_terms if t not in existing_keys and t in BudgetIO.key]
        invalid_terms = [t for t in budget_terms if t not in BudgetIO.key and t not in BudgetIO.key.inverse]

        valid_tup = [tup for tup in budget_terms if tup in existing_tup]  # existing tuples
        missing_tup = [tup for tup in budget_terms if tup not in existing_tup and tup in BudgetIO.key.inverse]

        # now combine existing valid keys and valid tuples, removing any duplicates

        valid_terms = set(valid_keys + [BudgetIO.key.inverse[tup][0] for tup in valid_tup])  # combine and remove duplicates
        missing_terms = set(missing_keys + [BudgetIO.key.inverse[tup][0] for tup in missing_tup])

        # generate the key
        key_subset = {key: BudgetIO.key[key] for key in valid_terms}
        
        # warn the user if some requested terms did not exist
        if len(key_subset) == 0: 
            warnings.warn('_parse_budget_terms(): No keys being returned; none matched.')

        if len(missing_terms) > 0: 
            warnings.warn('_parse_budget_terms(): Several terms were requested but the following could not be found: \
                {}'.format(missing_terms))

        if len(invalid_terms) > 0: 
            warnings.warn('_parse_budget_terms(): The following budget terms were requested but the following do not exist: \
                {}'.format(invalid_terms))

        # TODO - fix warning messages for the wakes

        return key_subset 


    def _get_inflow(self, offline=False, wInflow=False): 
        """
        Calls the appropriate functions in inflow.py to retrieve the inflow profile for the corresponding flow. 

        Arguments
        ---------
        offline (bool) : if True, uses the target inflow profile prescribed by initialize.F90. Default (False) reads
            the inflow profile from the first x-index of the domain and average over y. 
        wInflow (bool) : if True, returns an array of w-inflow velocities. Default (False) only returns u, v. 
        
        Returns
        -------
        u (array) : [nz x 1] array of u-velocities as a function of height
        v (array) : [nz x 1] array of v-velocities as a function of height
        w (array) : [nz x 1] array of w-velocities as a function of height. Nominally this is all zero. 

        """
        
        # load using InflowParser: 

        if offline: 
            if self.associate_nml: 
                u, v = inflow.InflowParser.inflow_offline(**dict(self.input_nml['AD_coriolisinput']), zLine=self.zLine)
            
            # reading from the budgets
            else: 
                warnings.warn('_get_inflow: Requested offline inflow, but namelist not associated. Trying online.')
                u, v = inflow.InflowParser.inflow_budgets(self)
                
        else: 
            u, v = inflow.InflowParser.inflow_budgets(self) 

        # return requested information 

        if wInflow: 
            # If this is not nominally zero, then this will need to be fixed. 
            w = np.zeros(self.zLine.shape)
            return np.array([u, v, w])

        else: 
            return np.array([u, v])

    
    def calc_wake(self, offline=False, wInflow=False, overwrite=False):
        """
        Computes the wake deficit by subtracting the target inflow from the flow field. 
        # TODO - right now this must compute at minimum uwake and vwake. Fix?  

        Arguments
        ---------
        see _get_inflow()
        overwrite (bool) : re-computes wakes if they are already read in. 

        Returns
        -------
        None (updates self.budget[] with 'uwake' and 'vwake' keys)

        """ 

        target_terms = ['uwake', 'vwake']
        req_terms = ['ubar', 'vbar']  # required budget terms
        if wInflow: 
            target_terms.append('wwake')
            req_terms.append('wbar') # might also need w

        # check to see if the terms exist already
        if all(t in self.budget.keys() for t in target_terms): 
            if not overwrite: 
                warnings.warn('Wake terms already computed, returning. To compute anyway, use keyword overwrite=True')
                return
        
        # Need mean velocity fields to be loaded
        if not all(t in self.budget.keys() for t in req_terms): 
            self.read_budgets(budget_terms=req_terms)

        # retrieve inflow
        if wInflow: 
            u, v, w = self._get_inflow(offline=offline, wInflow=wInflow)
            self.budget['wwake'] = self.budget['wbar'] - w[np.newaxis, np.newaxis, :]
        
        else: 
            u, v = self._get_inflow(offline=offline, wInflow=wInflow)

        self.budget['uwake'] = u[np.newaxis, np.newaxis, :] - self.budget['ubar']
        self.budget['vwake'] = v[np.newaxis, np.newaxis, :] - self.budget['vbar']

        if self.verbose: 
            print("calc_wake(): Computed wake velocities. ")

    
    def slice(self, budget_terms=None, 
              field=None, field_terms=None, 
              sl=None, keys=None, tidx=None, 
              xlim=None, ylim=None, zlim=None, 
              overwrite=False, round_extent=False): 
        """
        Returns a slice of the requested budget term(s) as a dictionary. 

        Parameters
        ----------
        budget_terms : list or string
            budget term or terms to slice from. If None, expects a value for `field` or `sl`
        field : array-like or dict of arraylike
            fields similar to self.field[] or self.budget[]
        field_terms: list
            read fields from read_fields(). 
        sl : slice from self.slice()
            dictionary of fields to be sliced into again. 
        keys : list 
            fields in slice `sl`. Keys to slice into from the input slice `sl`
        tidx : int
            time ID to read budgets from, see read_budgets(). Default None
        xlim, ylim, zlim : tuple
            in physical domain coordinates, the slice limits. If an integer is given, 
            then the dimension of the slice will be reduced by one. If None is given 
            (default), then the entire domain extent is sliced. 
        overwrite : bool
            Overwrites loaded budgets, see read_budgets(). Default False
        round_extent : bool
            Rounds extents to the nearest integer. Default False
        
        Returns
        -------
        slices : dict
            dictionary organized with all of the sliced fields, keyed by the budget name, 
            and additional keys for the slice domain 'x', 'y', 'z', and 'extent'
        """

        if sl is None: 
            xid, yid, zid = self.get_xids(x=xlim, y=ylim, z=zlim, return_none=True, return_slice=True)
            xLine = self.xLine
            yLine = self.yLine
            zLine = self.zLine
        else: 
            xid, yid, zid = self.get_xids(x=xlim, y=ylim, z=zlim, 
                                          x_ax=sl['x'], y_ax=sl['y'], z_ax=sl['z'], 
                                          return_none=True, return_slice=True)
            xLine = sl['x']
            yLine = sl['y']
            zLine = sl['z']

        slices = {'keys': []}  # build from empty dict
        preslice = {}

        if field_terms is not None: 
            # read fields
            self.read_fields(field_terms=field_terms, tidx=tidx)
            field = self.field

        # parse what field arrays to slice into
        if field is not None:             
            if isinstance(field, dict): 
                # iterate through dictionary of fields
                if keys is None: 
                    keys = field.keys()
                preslice = field
            else: 
                preslice = {'field': field}
                keys = ['field']

        elif budget_terms is not None: 
            # read budgets
            keys = self._parse_budget_terms(budget_terms)
            self.read_budgets(budget_terms=keys, tidx=tidx, overwrite=overwrite)
            preslice = self.budget

        elif sl is not None: 
            preslice = sl
            # parse keys: 
            if keys is None: 
                keys = sl['keys']
            elif type(keys) != list: 
                keys = [keys]
                
        else: 
            warnings.warn("BudgetIO.slice(): either budget_terms= or field= must be initialized.")
            return None
        
        # slice into arrays; now accommodates 3D arrays
        dims = (len(xLine), len(yLine), len(zLine))
        for key in keys: 
            slices[key] = np.squeeze(np.reshape(preslice[key], dims)[xid, yid, zid])
            slices['keys'].append(key)

        # also save domain information
        slices['x'] = xLine[xid]
        slices['y'] = yLine[yid]
        slices['z'] = zLine[zid]
        
        # build and save the extents, either in 1D, 2D, or 3D
        ext = []
        for term in ['x', 'y', 'z']: 
            if slices[term].ndim > 0:  # if this is actually a slice (not a number), then add it to the extents
                ext += [np.min(slices[term]), np.max(slices[term])]
        
        if round_extent: 
            slices['extent'] = np.array(ext).round()
        else: 
            slices['extent'] = np.array(ext)
        
        return slices


    def get_xids(self, **kwargs): 
        """
        Translates x, y, and z limits in the physical domain to indices based on self.xLine, self.yLine, and self.zLine

        Parameters
        ---------
        x, y, z : float or iterable (tuple, list, etc.) 
            Physical locations to return the nearest index 
        return_none : bool
            If True, populates output tuple with None if input is None. Default False. 
        return_slice : bool 
            If True, returns a tuple of slices instead a tuple of lists. Default False. 

        Returns
        -------
        xid, yid, zid : list or tuple of lists 
            Indices for the requested x, y, z, args in the order: x, y, z. 
            If, for example, y and z are requested, then the returned tuple will have (yid, zid) lists. 
            If only one value (float or int) is passed in for e.g. x, then an integer will be passed back in xid. 
        """

        if not self.associate_grid: 
            raise(AttributeError('No grid associated. '))

        # set up this way in case we want to introduce an offset later on (i.e. turbine-centered coordinates)
        if 'x_ax' not in kwargs or kwargs['x_ax'] is None: 
            kwargs['x_ax'] = self.xLine  
        if 'y_ax' not in kwargs or kwargs['y_ax'] is None: 
            kwargs['y_ax'] = self.yLine  
        if 'z_ax' not in kwargs or kwargs['z_ax'] is None: 
            kwargs['z_ax'] = self.zLine  

        return get_xids(**kwargs)
    
    
    def xy_avg(self, budget_terms=None, zlim=None, **slice_kwargs): 
        """
        xy-averages requested budget terms
        
        Parameters
        budget_terms : list
            Budget terms, see _parse_budget_terms()
        zlim : array-like
            z-limits for slicing
        slice_kwargs : Any
            Additional keyword arguments, see BudgetIO.slice()
        """
        _slice_kwargs = dict(xlim=None, ylim=None, zlim=zlim)
        _slice_kwargs.update(slice_kwargs)
        tmp = self.slice(budget_terms=budget_terms, **_slice_kwargs)
        
        ret = {}
        for key in tmp['keys']: 
            ret[key] = np.mean(tmp[key], (0, 1))
            
        ret['keys'] = tmp['keys']
        ret['z'] = tmp['z']
        return ret
    

    def unique_tidx(self, return_last=False, search_str='Run{:02d}.*_t(\d+).*.out'): 
        """
        Pulls all the unique tidx values from a directory. 
        
        Parameters 
        ----------
        return_last : bool
            If True, returns only the largest value of TIDX. Default False. 

        Returns
        -------
        t_list : array
            List of unique time IDs (TIDX)
        return_last : bool, optional
            if True, returns only the last (largest) entry. Default False
        search_str : regex, optional
            Regular expression for the search string and capture groups
        """

        if not self.associate_padeops:
            return None  # TODO - is this lost information? is it useful information? 
            
        # retrieves filenames and parses unique integers, returns an array of unique integers
        filenames = os.listdir(self.dir_name)
        runid = self.runid
        
        # searches for the formatting *_t(\d+)* in all filenames
        t_list = [int(re.findall(search_str.format(runid), name)[0]) 
                  for name in filenames 
                  if re.findall(search_str.format(runid), name)]
        
        if len(t_list) == 0: 
            raise FileNotFoundError('unique_tidx(): No files found')
        
        t_list.sort()
        
        if return_last: 
            return t_list[-1]
        else: 
            return np.unique(t_list)

    
    def unique_budget_tidx(self, return_last=True): 
        """
        Pulls all the unique tidx values from a directory. 
        
        Parameters
        ----------
        return_last : bool
            If False, returns only the largest TIDX associated with budgets. Else, 
            returns an entire list of unique tidx associated with budgets. Default True

        Returns
        -------
        t_list : array
            List of unique budget time IDs (TIDX)
        return_last : bool, optional
            if True, reutnrs only the last (largest) entry. Default True
        """

        # TODO: fix for .npz
        if not self.associate_padeops: 
            return None 
            
        return self.unique_tidx(return_last=return_last, search_str='Run{:02d}.*budget.*_t(\d+).*')
        
        
    def unique_times(self, return_last=False): 
        """
        Reads the .out file of each unique time and returns an array of [physical] times corresponding
        to the time IDs from unique_tidx(). 

        Parameters 
        ----------
        return_last : bool
            If True, returns only the largest time. Default False. 

        Returns
        -------
        times : array
            list of times associated with each time ID in unique_tidx()
        """
        
        # TODO: fix for .npz
        if not self.associate_padeops: 
            return None  

        times = []; 
        
        if return_last:  # save time by only reading the final TIDX
            tidx = self.unique_tidx(return_last=return_last)
            fname = os.path.join(self.dir_name, "Run{:02d}_info_t{:06d}.out".format(self.runid, tidx))
            t = np.genfromtxt(fname, dtype=None)[0]
            return t
        
        for tidx in self.unique_tidx(): 
            fname = os.path.join(self.dir_name, "Run{:02d}_info_t{:06d}.out".format(self.runid, tidx))
            t = np.genfromtxt(fname, dtype=None)[0]
            times.append(t)

        return np.array(times)

    
    def last_budget_n(self): 
        """
        Pulls all unique n from budget terms in a directory and returns the largest value. 
        """

        # TODO: fix for .npz
        if not self.associate_padeops: 
            return None 
            
        return self.unique_tidx(return_last=True, search_str='Run{:02d}.*_n(\d+).*')
    
    
    def existing_budgets(self): 
        """
        Checks file names for which budgets were output.  
        """
        filenames = os.listdir(self.dir_name)

        if self.associate_padeops: 
            runid = self.runid
            # capturing *_budget(\d+)* in filenames
            budget_list = [int(re.findall('Run{:02d}.*_budget(\d+).*'.format(runid), name)[0]) 
                           for name in filenames 
                           if re.findall('Run{:02d}.*_budget(\d+).*'.format(runid), name)]
        else: 
            if self.associate_npz: 
                filename = self.dir_name + os.sep + self.filename_budgets + '.npz'
                with np.load(filename) as npz: 
                    t_list = npz.files  # load all the budget filenames
            if self.associate_mat: 
                filename = self.dir_name + os.sep + self.filename_budgets + '.mat'
                ret = loadmat(filename)
                t_list = [key for key in ret if key[0] != '_']  # ignore `__header__`, etc. 
            
            budget_list = [BudgetIO.key[t][0] for t in t_list]

        if len(budget_list) == 0: 
            warnings.warn('existing_budgets(): No associated budget files found. ')
        
        if 0 in budget_list: 
            budget_list.append(5)  # wake budgets can be recovered from mean budgets

        return list(np.unique(budget_list))
    
    
    def existing_terms(self, budget=None, include_wakes=False): 
        """
        Checks file names for a particular budget and returns a list of all the existing terms.  

        Arguments 
        ---------
        budget (integer) : optional, default None. If provided, searches a particular budget for existing terms. 
            Otherwise, will search for all existing terms. `budget` can also be a list of integers. 
            Budget 0: mean statistics
            Budget 1: momentum
            Budget 2: MKE
            Budget 3: TKE
            Budget 5: Wake deficit
        include_wakes (bool) : Includes wakes in the returned budget terms if True, default False. 

        Returns
        -------
        t_list (list) : list of tuples of budgets found

        """

        t_list = []
        
        # if no budget is given, look through all saved budgets
        if budget is None: 
            budget_list = self.existing_budgets()
        
        else: 
            # convert to list if integer is given
            if type(budget) != list: 
                budget_list = [budget]
            else: 
                budget_list = budget

        # find budgets by name matching with PadeOps output conventions
        if self.associate_padeops: 

            filenames = os.listdir(self.dir_name)
            runid = self.runid
            
            tup_list = []
            # loop through budgets
            for b in budget_list: 
                # capturing *_term(\d+)* in filenames
                terms = [int(re.findall('Run{:02d}_budget{:01d}_term(\d+).*'.format(runid, b), name)[0]) 
                        for name in filenames if 
                        re.findall('Run{:02d}_budget{:01d}_term(\d+).*'.format(runid, b), name)]
                tup_list += [((b, term)) for term in set(terms)]  # these are all tuples
                
                # wake budgets: 
                wake_budgets = (1, 2, 3)
                if include_wakes and b == 5:  
                    terms = [int(re.findall('Run{:02d}_budget{:01d}_term(\d+).*'.format(runid, 0), name)[0]) 
                            for name in filenames if 
                            re.findall('Run{:02d}_budget{:01d}_term(\d+).*'.format(runid, 0), name)]  # read from mean budgets

                    tup_list += [((b, term)) for term in wake_budgets if term in terms]
            
            # convert tuples to keys
            t_list = [BudgetIO.key.inverse[key][0] for key in tup_list]
        # find budgets matching .npz convention in write_npz()
        else: 
            if self.associate_npz: 
                filename = self.dir_name + os.sep + self.filename_budgets + '.npz'
                with np.load(filename) as npz: 
                    all_terms = npz.files
                
            elif self.associate_mat: 
                filename = self.dir_name + os.sep + self.filename_budgets + '.mat'
                ret = loadmat(filename)
                all_terms = [key for key in ret if key[0] != '_']  # ignore `__header__`, etc. 

            else: 
                raise AttributeError('existing_budgets(): How did you get here? ')

            if budget is None:  # i.e. requesting all budgets
                return all_terms  # we can stop here without sorting through each budget
            
            tup_list = [BudgetIO.key[t] for t in all_terms]  # list of associated tuples
            t_list = []  # this is the list to be built and returned

            for b in budget_list: 
                t_list += [tup for tup in tup_list if tup[0] == b]

        # else: 
        if len(t_list) == 0: 
            warnings.warn('existing_terms(): No terms found for budget ' + str(budget))

        return t_list
    

    def Read_x_slice(self, xid, label_list=['u'], tidx_list=[]):
        """
        Reads slices of dumped quantities at a time ID or time IDs. 
        
        Arguments
        ---------
        xid (int) : integer of xid dumped by initialize.F90. NOTE: Fortran indexing starts at 1. 
        label_list (list) : list of terms to read in. Available is typically: "u", "v", "w", and "P" (case-sensitive)
        tidx_list (list) : list of time IDs. 
        
        Returns
        -------
        sl (dict) : formatted dictionary similar to BudgetIO.slice()
        """
        
        sl = {}
        if type(label_list)==str: 
            label_list = [label_list]
        
        for tidx in tidx_list:  
            for lab in label_list: 
                fname = "{:s}/Run{:02d}_t{:06d}_{:s}{:05d}.pl{:s}".format(self.dir_name, self.runid, tidx, 'x', xid, lab)

                key_name = "{:s}_{:d}".format(lab, tidx)
                sl[key_name] = np.fromfile(
                    fname, dtype=np.dtype(np.float64), count=-1).reshape((self.ny,self.nz), order='F')
            
        sl['x'] = self.xLine[[xid-1]]
        sl['y'] = self.yLine
        sl['z'] = self.zLine

        # build and save the extents, either in 1D, 2D, or 3D
        ext = []
        for term in ['x', 'y', 'z']: 
            if len(sl[term]) > 1:  # if this is actually a slice (not a number), then add it to the extents
                ext += [np.min(sl[term]), np.max(sl[term])]

        sl['extent'] = ext

        return sl

    
    def Read_y_slice(self, yid, label_list=['u'], tidx_list=[]):
        """
        Reads slices of dumped quantities at a time ID or time IDs. 
        
        Arguments
        ---------
        yid (int) : integer of yid dumped by initialize.F90
        label_list (list) : list of terms to read in. Available is typically: "u", "v", "w", and "P" (case-sensitive)
        tidx_list (list) : list of time IDs. 
        
        Returns
        -------
        sl (dict) : formatted dictionary similar to BudgetIO.slice()
        """
        
        sl = {}
        if type(label_list)==str: 
            label_list = [label_list]
        
        for tidx in tidx_list:  
            for lab in label_list: 
                fname = "{:s}/Run{:02d}_t{:06d}_{:s}{:05d}.pl{:s}".format(self.dir_name, self.runid, tidx, 'y', yid, lab)

                key_name = "{:s}_{:d}".format(lab, tidx)
                sl[key_name] = np.fromfile(
                    fname, dtype=np.dtype(np.float64), count=-1).reshape((self.nx,self.nz), order='F')
            
        sl['x'] = self.xLine
        sl['y'] = self.yLine[[yid-1]]
        sl['z'] = self.zLine

        # build and save the extents, either in 1D, 2D, or 3D
        ext = []
        for term in ['x', 'y', 'z']: 
            if len(sl[term]) > 1:  # if this is actually a slice (not a number), then add it to the extents
                ext += [np.min(sl[term]), np.max(sl[term])]

        sl['extent'] = ext

        return sl
    
    
    def Read_z_slice(self, zid, label_list=['u'], tidx_list=[]):
        """
        Reads slices of dumped quantities at a time ID or time IDs. 
        
        Arguments
        ---------
        zid (int) : integer of zid dumped by initialize.F90
        label_list (list) : list of terms to read in. Available is typically: "u", "v", "w", and "P" (case-sensitive)
        tidx_list (list) : list of time IDs. 
        
        Returns
        -------
        sl (dict) : formatted dictionary similar to BudgetIO.slice()
        """
        
        sl = {}
        if type(label_list)==str: 
            label_list = [label_list]
        
        for tidx in tidx_list:  
            for lab in label_list: 
                fname = "{:s}/Run{:02d}_t{:06d}_{:s}{:05d}.pl{:s}".format(self.dir_name, self.runid, tidx, 'z', zid, lab)

                key_name = "{:s}_{:d}".format(lab, tidx)
                sl[key_name] = np.fromfile(
                    fname, dtype=np.dtype(np.float64), count=-1).reshape((self.nx,self.ny), order='F')
            
        sl['x'] = self.xLine
        sl['y'] = self.yLine
        sl['z'] = self.zLine[[zid-1]]

        # build and save the extents, either in 1D, 2D, or 3D
        ext = []
        for term in ['x', 'y', 'z']: 
            if len(sl[term]) > 1:  # if this is actually a slice (not a number), then add it to the extents
                ext += [np.min(sl[term]), np.max(sl[term])]

        sl['extent'] = ext

        return sl
    
    
    def _read_turb_file(self, prop, tid=None, turb=1, steady=True): 
        """
        Reads the turbine power from the output files 

        Arguments
        ---------
        prop (str) : property string name, either 'power', 'uvel', or 'vvel'
        tidx (int) : time ID to read turbine power from. Default: calls self.unique_tidx()
        turb (int) : Turbine number. Default 1
        steady (bool) : Averages results if True. If False, returns an array containing the contents of `*.pow`. 
        """
        if prop == 'power': 
            fstr = '/Run{:02d}_t{:06d}_turbP{:02}.pow'
        elif prop == 'uvel': 
            fstr = '/Run{:02d}_t{:06d}_turbU{:02}.vel'
        elif prop == 'vvel': 
            fstr = '/Run{:02d}_t{:06d}_turbV{:02}.vel'
        else: 
            raise ValueError("_read_turb_prop(): `prop` property must be 'power', 'uvel', or 'vvel'")
        
        if tid is None: 
            try: 
                tid = self.last_tidx
            except ValueError as e:   # TODO - Fix this!! 
                tid = self.unique_tidx(return_last=True)
        
        fname = self.dir_name + fstr.format(self.runid, tid, turb)
        if self.verbose: 
            print("\tReading", fname)
            
        ret = np.genfromtxt(fname, dtype=float)  # read fortran ASCII output file
        
        # for some reason, np.genfromtxt makes a size 0 array for length-1 text files. 
        # Hotfix: multiply by 1. 
        ret = ret*1  
        
        if steady: 
            return np.mean(ret)
        else: 
            return ret  # this is an array
        
        
    def read_turb_property(self, tidx, prop_str, turb=1, steady=None): 
        """
        Helper function to read turbine power, uvel, vvel. Calls self._read_turb_file() 
        for every time ID in tidx. 
        """
        
        if not self.associate_padeops:  # read from saved files
            if self.associate_mat: 
                fname = self.dir_name + os.sep + self.filename + '_metadata.mat'
                tmp = loadmat(fname)

            elif self.associate_npz: 
                fname = self.dir_name + os.sep + self.filename + '_metadata.npz'
                tmp = np.load(fname)
            else: 
                raise AttributeError('read_turb_property(): How did you get here? ')
            
            try: 
                return np.squeeze(tmp[f't{turb}_{prop_str}'])
            except KeyError as e: 
                raise e

        # else: read from padeops: 
        prop_time = []  # power array to return

        if tidx is None: 
            tidx = [self.last_tidx]  # just try the last TIDX by default
        elif tidx == 'all': 
            tidx = self.unique_tidx(search_str='Run{:02d}.*_t(\d+).*.pow')
        
        if not hasattr(tidx, '__iter__'): 
            tidx = np.atleast_1d(tidx)
            
        if steady == None: 
            if len(tidx) > 1: 
                steady = False  # assume if calling for more than 1 time ID, that steady is FALSE by default
            else: 
                steady = True
            
        for tid in tidx:  # loop through time IDs and call helper function
            prop = self._read_turb_file(prop_str, tid=tid, turb=turb, steady=steady)
            if type(prop) == np.float64:  # if returned is not an array, cast to an array
                prop = np.array([prop])
            prop_time.append(prop)

        prop_time = np.concatenate(prop_time)  # make into an array
        
        # only select unique values... for some reason some values are written twice once budgets start up
        _, prop_index = np.unique(prop_time, return_index=True)

        return prop_time[np.sort(prop_index)]  # this should make sure that n_powers = n_tidx


    def read_turb_power(self, tidx=None, **kwargs): 
        """
        Reads the turbine power files output by LES in Actuator Disk type 2 and type 5. 
        
        Parameters
        ----------
        tidx (iterable) : list or array of time IDs to load data. Default: self.last_tidx. 
            If tidx = 'all', then this calls self.unique_tidx()
        **kwargs() : see self._read_turb_file()
        """
        return self.read_turb_property(tidx, 'power', **kwargs)
    
    
    def read_turb_uvel(self, tidx=None, **kwargs): 
        """
        Reads turbine u-velocity. 
        
        See self.read_turb_power() and self._read_turb_file()
        """
        return self.read_turb_property(tidx, 'uvel', **kwargs)
    
    
    def read_turb_vvel(self, tidx=None, **kwargs): 
        """
        Reads turbine v-velocity
        
        See self.read_turb_power() and self._read_turb_file()
        """
        return self.read_turb_property(tidx, 'vvel', **kwargs)
    

if __name__ == "__main__": 
    """
    TODO - add unit tests to class
    """
    print("padeopsIO: No unit tests included yet. ")
