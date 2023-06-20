import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from joblib import dump, load
from datetime import datetime
import psutil
import os
from os import path
import time
import random
from typing import Optional
import tkinter as tk
from tkinter.messagebox import showinfo, showwarning
import gym
from gym import spaces
import ansys
from ansys.mapdl.core import launch_mapdl
from ansys.mapdl import reader as mapdl_reader


class FuselageActuatorsEnv(gym.Env):
    """
    ### Description
    The goal of this environment is to minimize the shape error of a fuselage by adjusting the forces exerted
    by actuators at all of the 18 positions around the lower circumference of the part. At each time step, add a force to each actuator location
    
    ### Observation Space
    Observations consist of the current position deviation of nodes from their target positions along the edge of the fuselage

    ### Action Space
    At each time step, the force at one of the 18 actuator locations can be set to a target value.
    Forces are applied all at once for the n actuators with the largest magnitude of force specified.

    a[0-17]: continuous number between -1000 and 1000. 
    Note: Action space is normalized into the interval [-1,1] and scaled by the environment
    
    ### Rewards
    At the end of each episode, max(1-error_new/error_initial, -1) is returned. 
    
    ### Episode termination
    Episodes terminate after exactly one step (one-shot simulation)

    ```
    env = gym.make('FuselageActuators-v12')
    ```
    """
    metadata = {'render_modes': ['human'],
                'modes': ['Train', 'Test', 'File', 'Surrogate']
    }

    def __init__(self, render_mode: Optional[str] = None, n_actuators=10, mode='Train', port=50056, file1=None, file2=None, record=False, seed=0):
        super(FuselageActuatorsEnv, self).__init__()
        # Process keyword arguments
        self.mode = mode
        self.render_mode = render_mode
        self.record = record
        self.port = port
        self.file1 = file1
        self.file2 = file2
        self.n_actuators = n_actuators
        self.surrogate = None
        random.seed(seed)
        # if self.mode=="File":
        #     assert is_instance(file1, str), "Must specify a file path for the starting positions"
        #     assert is_instance(file2, str), "Must specify a file path for the target positions"

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Box(-1, 1, shape = (18,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(177*2,), dtype=np.float32) # 177 deviations of nodal positions from their targets

        if self.mode != 'Surrogate':
            # Check if MAPDL server is active, and start it if it's not
            if not self._monitor_process('ansys'):
                self._launch_ansys()
        elif self.mode == 'Surrogate':
            self.surrogate = load(path.join(path.dirname(__file__), 'Surrogates', 'surrogate_likeDu_v22.joblib') )

        # Generate headers for recording
        self.h1= []
        self.h2= []
        self.h3= []
        for h in range(177):
            self.h1.append("initDev"+str(h+1))
            self.h3.append("finalDev"+str(h+1))
        for h in range(18):
            self.h2.append("Force"+str(h+1))
        
        # Record file location
        timestamp = datetime.now()
        timestampStr = timestamp.strftime("%Y%m%d-%H%M")
        folder = path.join(path.dirname(__file__), 'Recordings', 'FuselageActuators-v22', self.mode) 
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.recordPath= path.join(folder, timestampStr+".csv")
        
        
    def step(self, action):
        # Pick the ten largest forces, keep others at zero
        n = self.n_actuators
        # idx = (-abs(action)).argsort()[:n]
        # self.forces[idx] += action[idx]*1000 # Action space (-1,1) scaled to (-1000lb, 1000lb)

        self.forces += action*1000 # Action space (-1,1) scaled to (-1000lb, 1000lb)
        idx = (abs(action)).argsort()[:18-n]
        
        self.forces[idx] = 0
        # print(self.forces)
        
        if self.mode != 'Surrogate':
            # Check if MAPDL server is active, and restart it if it's not
            if not self._monitor_process('ansys'):
                self._launch_ansys()  

            # Run the Ansys simulation with forces
            self._run_ansys()
            # Get displacements from Ansys
            self.displacements = self._get_displacement()
            u = self.displacements.flatten()

            # Track 
            p_init = self.initPos[:,0:2].flatten()
            p_final = p_init + u
            p_target =self.targetPos[:,0:2].flatten()
            self.deviations = p_final - p_target
            # Assemble observation
            obs = self.deviations
            obs = obs.flatten()

        elif self.mode == 'Surrogate':
            # Calculate y and z components of the forces from desired magnitudes
            angles = np.linspace(12, -192, 18)
            self.forces_Y = self.forces*np.cos(np.deg2rad(angles))
            self.forces_Z = self.forces*np.sin(np.deg2rad(angles))
            # Predict deviations from surrogate model
            u = self.surrogate.predict(np.expand_dims(self.forces, axis=0)).flatten()
            self.displacements = u.reshape((-1,2))

            # Track 
            p_init = self.initPos[:,0:2].flatten()
            p_final = p_init + u
            p_target =self.targetPos[:,0:2].flatten()
            self.deviations = p_final - p_target
            # Assemble observation
            obs = self.deviations
            obs = obs.flatten()
        
        # Calculate the error
        self.error, self.maxDev = self._get_errors()
        
        # Terminate after one time step
        done = True

        # Calcualate the reward
        self.reward = max((1-self.error/self.error_old), -1)
        self.error_old=self.error
        self.maxDev_old=self.maxDev

        # Output info    
        info = {"initError":self.error_initial, "Error":self.error, "Forces":self.forces, "maxDev": self.maxDev, "initMaxDev": self.maxDev_initial}
        # self.render()

        # Record the interaction to csv file
        if self.record and self.mode != 'Surrogate':
            self._record()

        return np.array(obs, dtype=np.float32), self.reward, done, info

    def reset(self):
        # print("Resetting the environment")
        # Check if MAPDL server is active, and start it if it's not
        if self.mode != 'Surrogate':
            if not self._monitor_process('ansys'):
                self._launch_ansys()  

        # Set forces to zero
        self.forces = np.zeros(18, dtype=np.float32) 

        # Check mode
        if self.mode == 'Train' or self.mode == 'Test':

            # Select Ansys input file (randomly)
            folder = path.join(path.dirname(__file__), 'AnsysFiles', self.mode) 
            file = random.choice(os.listdir(folder))
            filepath = path.join(folder, file)
            print("Initial shape from", file.split('.')[0])

            # Parse the Ansys file
            with open(filepath, 'r') as f:
                text = f.read()
                new_text = text.split('/com,******************* SOLVE FOR LS 1 OF 1 ****************')
                self.setup_text = new_text[0]
                new_text = text.split('! *********** WB SOLVE COMMAND ***********')
                self.finish_text = new_text[1]
                f.close()

            # Load precalculated nodal positions
            folder = path.join(path.dirname(__file__), 'Shapes', self.mode) 
            file1 = file.split(".")[0] + ".npy"
            filepath = path.join(folder, file1)
            self.initPos = np.load(filepath)
            self.displacements = np.zeros((177,2))

            # Randomly select source for target positions
            folder = path.join(path.dirname(__file__), 'Shapes', self.mode) 
            file2 = random.choice(os.listdir(folder))
            while file1 == file2:   # make sure files are not the same
                file2 = random.choice(os.listdir(folder))
            # Load precalculated target positions
            filepath = path.join(folder, file2)
            print("Target shape from", file2.split('.')[0])
            self.targetPos = np.load(filepath)    
            self.deviations = self._get_deviation()
            self.initDev = self.deviations # for recording

        # File mode
        elif self.mode == 'File':
            # Set filepath to input from environment creation
            filepath = self.file2

            # Parse the Ansys file
            with open(filepath, 'r') as f:
                text = f.read()
                new_text = text.split('/com,******************* SOLVE FOR LS 1 OF 1 ****************')
                self.setup_text = new_text[0]
                new_text = text.split('! *********** WB SOLVE COMMAND ***********')
                self.finish_text = new_text[1]
                f.close()

            # Get target position from Ansys
            self._run_ansys()
            self.targetPos = self._get_initPos()

            # Set filepath to input from environment creation
            filepath = self.file1

            # Parse the Ansys file
            with open(filepath, 'r') as f:
                text = f.read()
                new_text = text.split('/com,******************* SOLVE FOR LS 1 OF 1 ****************')
                self.setup_text = new_text[0]
                new_text = text.split('! *********** WB SOLVE COMMAND ***********')
                self.finish_text = new_text[1]
                f.close()

            # Get initial position from Ansys
            self._run_ansys()
            self.initPos = self._get_initPos()
            self.displacements = self._get_displacement()
            self.deviations = self._get_deviation()
            self.initDev = self.deviations # for recording

        elif self.mode == 'Surrogate':
            # Load precalculated nodal positions
            folder = path.join(path.dirname(__file__), 'Shapes', 'Train') 
            file1 = random.choice(os.listdir(folder))
            filepath = path.join(folder, file1)
            self.initPos = np.load(filepath)
            self.displacements = np.zeros((177,2))
            # Randomly select source for target positions
            folder = path.join(path.dirname(__file__), 'Shapes', 'Train') 
            file2 = random.choice(os.listdir(folder))
            while file1 == file2:   # make sure files are not the same
                file2 = random.choice(os.listdir(folder))
            filepath = path.join(folder, file2)
            self.targetPos = np.load(filepath)  
            # Print info
            # print("Initial shape from", file1.split('.')[0])
            # print("Target shape from", file2.split('.')[0])

        # Get deviations
        self.deviations = self._get_deviation()

        # Assemble observation
        obs = self.deviations
        obs = obs.flatten()

        # Initialize the error
        self.error, self.maxDev = self._get_errors()
        self.error_best = self.error
        self.error_old = self.error
        self.error_initial = self.error
        self.maxDev_old = self.maxDev
        self.maxDev_initial = self.maxDev

        # Zero reward
        self.reward = 0
        
        # Display initial error
        # print("Initial Error =", self.error)

        return np.array(obs, dtype=np.float32)  # reward, done, info can't be included

    def render(self, mode='human'):
        print("Forces=", self.forces)
        print("Error=", self.error, " ==> Reward=", self.reward)

    def _record(self):
        # Build dataframes
        df1 = pd.DataFrame(self.initDev, self.h1).T
        df2 = pd.DataFrame(self.forces, self.h2).T
        df3 = pd.DataFrame(self.finalDev, self.h3).T
        # Join them together
        df = pd.concat([df1, df2, df3], axis=1)
        # Write csv file
        df.to_csv(self.recordPath, mode='a', header=not os.path.exists(self.recordPath))

    def _get_errors(self):
        # Needs to be called after getting observations so that data is up to date
        # Calculate error relative to perfect circle with r=288
        n = len(self.deviations)
        dev_total = np.sqrt(np.square(self.deviations[:177]) + np.square(self.deviations[177:]))
        max_e = max(dev_total) # maximum error
        mae = sum(np.abs(self.deviations))/n # mean absolute error
        rmse = np.sqrt(sum((self.deviations)**2)/n) # root mean squared error
        mse = sum((self.deviations)**2)/n # mean squared error
        se = sum((self.deviations)**2) # sum of squared errors
        return rmse, max_e

    def close (self):
        if self.mode != 'Surrogate':
            self.mapdl.exit() # close ANSYS

    def _get_obs(self):
        # Get displacements from simulation
        self.displacements = self._get_displacement()
        # Calculate deviations
        self.deviations = self._get_deviation()

        obs = self.deviations
        obs = obs.flatten() #np.expand_dims(obs, -1)
        return np.array(obs, dtype=np.float32)
        
    # Functions dealing with ANSYS and processing simulation results
    def _run_ansys(self):
          
        try:
            if not self._monitor_process('ansys'):
                self._launch_ansys()  
            # Clear solver memory
            self.mapdl.finish()
            self.mapdl.clear()
            #print("Ready to run")
            # Setup and run the simulation
            log1 = self.mapdl.input_strings(self.setup_text) # run setup
            log2 = self._set_actuator_forces(self.mapdl, self.forces) # apply forces
            log3 = self.mapdl.input_strings(self.finish_text) # complete solution
            self.result = self.mapdl.result # store result
            #print("Results are available")
        
        except:
            print("Exit Ansys and try to reconnect")
            
            try:
                self.mapdl.exit()
                print("Remote exit")
                time.sleep(10)
            except:
                print("No active Ansys process found. Wait and try to reconnect")
                time.sleep(10)
                
            i = 0
            while True:
                i += 1
                try: 
                    self._launch_ansys()
                    print("Sucessfully reconnected to Ansys on attempt", i)
                    print("Try running again")
                    log1 = self.mapdl.input_strings(self.setup_text) # run setup
                    print("Simulation setup complete")
                    log2 = self._set_actuator_forces(self.mapdl, self.forces) # apply forces
                    print("Applied forces")
                    log3 = self.mapdl.input_strings(self.finish_text) # complete solution
                    print("Solve finished")
                    self.result = self.mapdl.result # store result 
                    print("Results ready")         
                    
                    break

                except:
                    try:
                        print("Reconnect failed - remote exit again")
                        self.mapdl.exit()
                        time.sleep(10)
                    except:
                        time.sleep(10)
                        if i <=3:
                            print("Wait and try to reconnect again - attempt", i)
                        else:
                            print("Check Ansys license server connection")
                            # Create popup message
                            root = tk.Tk()
                            root.title('Warning')
                            root.geometry('300x150')
                            answer = showwarning(title='Warning',
                                                message='Check Ansys license server connection!')
                            if answer:
                                root.destroy()
                            time.sleep(5)

     
    def _get_stress(self):
        return self.mapdl.post_processing.nodal_eqv_stress()
        
    def _get_displacement(self):
        '''
        Get the displacement of nodes on the fuselage edge after forces have been applied. 
        The displacements are relative to the initial positions of the nodes.
        '''
        self.mapdl.cmsel(name='CM_FUSELAGE_EDGE') # select nodes on the edge of the fuselage
        displacements = self.mapdl.post_processing.nodal_displacement('ALL') # get displacements of nodes on the edge
        self.mapdl.allsel()
        return displacements[:,1:3]
        
    def _get_initPos(self):
        '''
        Get the initial positions of nodes on the fuselage edge before any forces are applied
        '''
        self.mapdl.cmsel(name='CM_FUSELAGE_EDGE') # select nodes on the edge of the fuselage
        initPos = self.mapdl.mesh.nodes # initial positions of nodes
        nnum = self.mapdl.mesh.nnum # corresponding node numbers
        self.mapdl.allsel()
        return initPos[:,1:3]
    
    def _get_deviation(self):
        '''
        Calculate the distance of the current node positions from their ideal positions
        ''' 
        finalPos = self.initPos + self.displacements
        deviations = finalPos - self.targetPos[:,0:2]
        return deviations.flatten()

    def _set_actuator_forces(self, mapdl, forces):
        # Calculate y and z components of the forces from desired magnitudes
        angles = np.linspace(12, -192, 18)
        self.forces_Y = forces*np.cos(np.deg2rad(angles))
        self.forces_Z = forces*np.sin(np.deg2rad(angles))
        
        # Apply the forces as surface force on selected elements
        for i in range(0,18):
            # Set x component of force (practically zero)
            mapdl.esel("s", "real", "", 27+3*i)
            mapdl.sfe("all", 1, "pres", 1, 2.24808943074769e-009)
            # Set y component of force
            mapdl.esel("s", "real", "", 28+3*i)
            mapdl.sfe("all", 1, "pres", 1, self.forces_Y[i])
            # Set z component of force
            mapdl.esel("s", "real", "", 29+3*i)
            mapdl.sfe("all", 1, "pres", 1, self.forces_Z[i])
        mapdl.esel("all")   # make sure everything is selected before running solve 
        
        # Run the solution
        mapdl._run("/nopr")
        mapdl.run("/gopr")
        mapdl.run("nsub,1,1,1")
        mapdl.time(1.)
        mapdl.outres("erase")
        mapdl.outres("all", "none")
        mapdl.outres("nsol", "all")
        mapdl.outres("rsol", "all")
        mapdl.outres("eangl", "all")
        mapdl.outres("etmp", "all")
        mapdl.outres("veng", "all")
        mapdl.outres("strs", "all")
        mapdl.outres("epel", "all")
        mapdl.outres("eppl", "all")
        mapdl.outres("cont", "all")

    def _monitor_process(self, processName):
        flag = False
        #Iterate over the all the running process
        for proc in psutil.process_iter():
            try:
                # Check if process name contains the given name string.
                if processName.lower() in proc.name().lower():
                    flag = True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return flag

    def _launch_ansys(self):
        # Launch ANSYS
        try:
            n_cpu = psutil.cpu_count(logical=False)
            self.mapdl = launch_mapdl(loglevel='ERROR', verbose=False, nproc=n_cpu, cleanup_on_exit=True, override=True) 
            print(self.mapdl)
            print("Running on", n_cpu, "processors")
        except:
            n_cpu=min(4, n_cpu) #license sometimes won't let me use more than 4 processors?
            self.mapdl = launch_mapdl(loglevel='ERROR', verbose=False, port=self.port, nproc=n_cpu, cleanup_on_exit=True, override=True) 
            print(self.mapdl)
            print("Running on", n_cpu, "processors")