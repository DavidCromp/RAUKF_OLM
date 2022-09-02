# Import Python SINGLE implementation
from OLM import OLM_Single
# Import hoc interpreter (for SINGLE and FULL)
from neuron import h
import numpy as np
# Import FULL model recording functions
from multicompartment.record import record_V_cai_ica_dend,\
    record_V_and_Cai, record_soma
# Import SINGLE model recording functions
from comp_opt_essential.record_1comp import set_up_full_recording
# Import plotting utility
import matplotlib.pyplot as plt
# Import RAUKF/UKF
from ukf import UkfEstimator
# Import Time estimated progress bars
from tqdm import tqdm
# Import random choose function
from random import choice
# Used to cycle stimulation
from itertools import cycle


# Whether estimating FULL model (True) or SINGLE (False)
multi=False;

if multi:
    h.load_file("./multicompartment/init_model_tweaked.hoc")
else:
    h.load_file("./comp_opt_essential/init_1comp.hoc")
    h.soma.ena=90
    h.soma.gna_Nasoma = 48.472e-4
    h.soma.gbar_Ikdrf = 43.343e-4
    h.soma.gbar_IM    = 0.19082e-4
    h.soma.gbar_Ika   = 73.062e-4

# Disable CVODE integration
h.cvode.active(False)

# Option for OU process injection
# gf = h.Gfluct2(h.soma(0.5))

# Ensure same timestep as used in NEURON
dt=h.dt

# Generates stimulation protocol given array of steps to used. Will
# cycle through given step values nstim times over time_scale seconds
def genStim(Is=[30],nstim=10,time_scale=0.5):
    tsig = 1e3*time_scale
    t  = np.arange(0,(500+tsig)+dt*2,dt)
    I  = np.full(t.shape,4.)
    I += np.random.normal(0,5,I.shape)
    Is = cycle(Is)
    for n in range(nstim):
        t1 = int((250+tsig/nstim*(n))/dt)
        t2 = int((250+tsig/nstim*(n+1))/dt)
        I[t1:t2] += next(Is)
    return t,I*1e-9

# Generate stimulation protocol as used in the main article
ts,I  = genStim([30,60,90,0,-30,-60,-90],nstim=28,time_scale=5)

# instantiate recording globally
# record from either FULL or SINGLE model respectively
rec=None
if multi:
    rec    = record_soma(h,0)
else:
    rec    = set_up_full_recording(h,False)

# Function to run FULL or SINGLE model with current I (protocol)
def runNeuron():
    Iinj = h.Vector(I*1e6)
    h.tstop = ts[-1]
    h.IClamp[0].amp=0.
    h.IClamp[1].delay=0
    h.IClamp[1].dur=1e9
    Iinj.play(h.IClamp[1]._ref_amp,h.dt)

    h.finitialize(-74)
    h.fcurrent()

    while (h.t<h.tstop):
        h.fadvance()
        
runNeuron()

# Generate observation of FULL or SINGLE with noise
obs = np.array([np.array(rec[0])])[:,:].T
obs += np.random.normal(0,3**0.5,size=obs.shape)

# List of states to estimate, full list of states defined in OLM.py
theta_keys = ["g_M",
              "g_kdrf",
              "g_ka",
              "g_nasoma"
              ]

# Run estimation of states above, optimalInit determined what the
# initial value should be. If True it is the same as the SINGLE value,
# otherwise it is modified as described in the main article
def runEst(optimalInit=False):
    # Create a SINGLE model
    nrn = OLM_Single()
    nrn.P[0,9]=90
    nrn.dt=0.025
    
    # Get value of all states wanting to be estimated
    theta = [list(nrn.p.keys()).index(param) for param in theta_keys]

    # Initial state vector for Optimal Init
    x_ = [-74,
          0,
          0,
          1,
          0.5,
          0,
          0.75,
          0,
          0,]
    x_.extend([nrn.p[t] for t in theta_keys])

    # Intial state vector for Poor Init
    x0 = [-74,
          0,
          0,
          1,
          0.5,
          0,
          0.75,
          0,
          0,]
    x0.extend([10**np.floor(np.log10(nrn.p[t])) for t in theta_keys])

    x0 = np.array([x0 if optimalInit else x_])

    # Process covariance estimates along diagonal
    P_ = [
        1e-4,
        1e-4,
        1e-4,
        1e-4,
        1e-4,
        1e-4,
        1e-4,
        1e-4,
        1e-4,
    ]
    P_.extend([10**(np.floor(np.log10(nrn.p[t]))*2) for t in theta_keys])

    # State covariance estimates along diagonal
    Q_ = [
        1e-8,
        1e-8,
        1e-8,
        1e-8,
        1e-8,
        1e-8,
        1e-8,
        1e-8,
        1e-8,
    ]
    Q_.extend([10**(np.round(np.log10(nrn.p[t]))*4) for t in theta_keys])

    P0 = np.diag(P_)
    Q0 = np.diag(Q_)
    # Observation noise, expected (actual noise is 3)
    R0 = np.diag(np.array([100]))

    # Reformat protocl I to interpretable format for ukf
    I_=np.array([I]).T

    # Produce estimator, robust determines if to use RAUKF or UKF (False)
    ukf = UkfEstimator(nrn,obs,I_,
                       theta,x0,P0,Q0,R0,kappa=0,sigma=0.5,robust=True)
    # Set bias of response when fault is detected in estimation. Set
    # as per parameter exploration completed in Azzalini (2022)
    ukf.a=10
    ukf.b=1
    # Run estimation
    x, P = ukf.run_estimation(resample=True,int_factor=1)
    # From results get last 100 samples of estimates of theta_keys and
    # take their mean
    newp = np.mean(np.reshape(x[~np.isnan(x)],(-1,x0.shape[1]))[-100:,9:],axis=0)
    return x,newp,P,ukf

# Multiprocessing, why wait longer
from pathos.multiprocessing import ProcessingPool as Pool
p = Pool(3)
# Run estimation on observation for both Poor (False) and Optimal (True)
est1,est2=p.map(runEst,[False,True])

# Extract tuples
x1,p1,P1,_ = est1
x2,p2,P2,_ = est2

# Extract diagonals of covariances
P1 = np.array([np.sqrt(np.diag(P1[i:i+P1.shape[1]])) for i in range(0, P1.shape[0], P1.shape[1])])
P2 = np.array([np.sqrt(np.diag(P2[i:i+P2.shape[1]])) for i in range(0, P2.shape[0], P2.shape[1])])

# Create a new SINGLE model
nrn = OLM_Single()

# extract SINGLE values
p_ = np.array([nrn.p[key] for key in theta_keys])

# Run SINGLE model with given parameters newp replacing theta_keys
def runPyWithP(newp=p_):
    t_nrn = OLM_Single()
    t_nrn.dt = 0.025
    for param,key in zip(newp,theta_keys):
        t_nrn.p[key] = param
    t_nrn.P = np.array([[v for v in t_nrn.p.values()]])

    ks = np.arange(1,I.shape[0])
    xs = np.zeros((I.shape[0],t_nrn.x.shape[1]))
    xs[[0]] = t_nrn.x
    p = t_nrn.P
    for k in tqdm(ks):
        xs[[k]] = t_nrn.forward(xs[[k-1]],I[k],p,int_factor=1)
    return xs
# Run SINGLE model with RAUKF Poor and Optimal, and SINGLE values for theta_keys
x_1, x_2, x__ = p.map(runPyWithP,[p1,p2,p_])
# Extract V
x_1 = x_1[:,0]
x_2 = x_2[:,0]
x__ = x__[:,0]
# Terminate multiprocessing pool
p.close()


## Figure Generation

# Whether to save figures
saveFig=False
# What file type to save them as
filetype='.svg'

plt.rcParams['font.size'] = 8
plt.rcParams['figure.dpi'] = 600
plt.rcParams['lines.linewidth'] = 0.75
plt.rcParams['legend.loc'] = 'upper right'
figsize = (18/2.54,18/2.54/((1+5**0.5)/2))
figsqar = (18/2.54,18/2.54)
# Colours used for theta_keys, must be same size as theta_keys
cs = [
    "#F21818",
    "#15A743",
    "#0952B1",
    "#ECE646",
    "#d0d"
]

# Used to modify file names based on if observing FULL or SINGLE
full="_Full" if multi else ""
model="FULL" if multi else "SINGLE"

plt.figure(figsize=figsize)
for x1_,p1_,x2_,p2_,o,c,t in zip(x1[:-1,9:].T,P1[:-1,9:].T,
                                 x2[:-1,9:].T,P2[:-1,9:].T,
                                 p_.T,cs,theta_keys):
    plt.plot(ts,x1_,color=c,label=t)
    plt.fill_between(ts,x1_-p1_,x1_+p1_,color=c,alpha=0.5)
    plt.plot(ts,x2_,color=c,alpha=0.5)
    plt.fill_between(ts,x2_-p2_,x2_+p2_,color=c,alpha=0.2,hatch='|')
    plt.hlines(o,ts[0],ts[-1],linestyle='--',color=c)
plt.xlabel("Time (ms)")
plt.ylabel("Conductance (nS)")
plt.legend()
if saveFig:
    plt.savefig(f"img/KF{full}_Est_{len(theta_keys)}_Param{filetype}")

plt.figure(figsize=figsize)
for x1_,p1_,x2_,p2_,o,c,t in zip(x1[:-1,9:].T,P1[:-1,9:].T,
                                 x2[:-1,9:].T,P2[:-1,9:].T,
                                 p_.T,cs,theta_keys):
    if t != 'g_M':
        continue
    plt.plot(ts,x1_,color=c,label=t)
    plt.fill_between(ts,x1_-p1_,x1_+p1_,color=c,alpha=0.5)
    plt.plot(ts,x2_,color=c,alpha=0.5)
    plt.fill_between(ts,x2_-p2_,x2_+p2_,color=c,alpha=0.2,hatch='|')
    plt.hlines(o,ts[0],ts[-1],linestyle='--',color=c)
plt.xlabel("Time (ms)")
plt.ylabel("Conductance (nS)")
plt.legend()
if saveFig:
    plt.savefig(f"img/KF{full}_Est_{len(theta_keys)}_Param_GM{filetype}")

plt.figure(figsize=figsize)
plt.plot(ts,obs[:-1],color='k',alpha=0.5)
plt.plot(ts,np.array(rec[0])[:-1],color='k',label=f'{model}')
plt.plot(ts,x_1,color=cs[0],label='Optimal Initial')
plt.plot(ts,x_2,color=cs[1],label='Poor Initial')
if multi:
    plt.plot(ts,x__,color=cs[2],label='SINGLE')
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.legend()
if saveFig:
    plt.savefig(f"img/KF{full}_Perf_Over_{len(theta_keys)}_param{filetype}")

plt.figure(figsize=figsize)
plt.subplot(211)
plt.plot(ts,obs[:-1],color='k',alpha=0.5)
plt.plot(ts,np.array(rec[0])[:-1],color='k',label=f'{model}')
plt.plot(ts,x_1,color=cs[0],label='Optimal Initial')
plt.plot(ts,x_2,color=cs[1],label='Poor Initial')
if multi:
    plt.plot(ts,x__,color=cs[2],label='SINGLE')
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.legend()
plt.subplot(212)
plt.xlabel('Time (ms)')
plt.ylabel(r'Current $pA$')
plt.plot(ts,I*1e9,color='k',label=r'$I_{stim}$')
plt.legend()
plt.tight_layout()
if saveFig:
    plt.savefig(f"img/KF{full}_Perf_Over_{len(theta_keys)}_param_and_Istim{filetype}")

plt.figure(figsize=figsize)
plt.plot(ts,obs[:-1],color='k',alpha=0.5)
plt.plot(ts,np.array(rec[0])[:-1],color='k',label=f'{model}')
plt.plot(ts,x_1,color=cs[0],label='Optimal Initial')
plt.plot(ts,x_2,color=cs[1],label='Poor Initial')
if multi:
    plt.plot(ts,x__,color=cs[2],label='SINGLE')
plt.xlim(200,250+(5000)/28*7)
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.legend()
if saveFig:
    plt.savefig(f"img/KF{full}_Perf_Zoom_Over_{len(theta_keys)}_param{filetype}")

plt.figure(figsize=figsize)
plt.subplot(221)
plt.plot(ts,obs[:-1],color='k',alpha=0.5)
plt.plot(ts,np.array(rec[0])[:-1],color='k',label=f'{model}')
xl = plt.xlim()
yl_ = plt.ylim()
yl = (yl_[0],yl_[1] + 10)
plt.ylim(yl)
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.legend()

plt.subplot(222)
plt.plot(ts,x_1,color=cs[0],label='Optimal Initial')
plt.xlim(xl)
plt.ylim(yl)
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.legend()

plt.subplot(223)
plt.plot(ts,x_2,color=cs[1],label='Poor Initial')
plt.xlim(xl)
plt.ylim(yl)
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.legend()
plt.tight_layout()
if multi:
    plt.subplot(224)
    plt.plot(ts,x__,color=cs[2],label='SINGLE')
    plt.xlim(xl)
    plt.ylim(yl)
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential (mV)")
    plt.legend()

if saveFig:
    plt.savefig(f"img/KF{full}_Perf_Subplot_{len(theta_keys)}_param{filetype}")

print(",Optimized Initial, Final, % Diff, Poor Initial, Final, % Diff, Actual Value")
for p1_,p2_,p__,t_ in zip(p1,p2,p_,theta_keys):
    print(f"{t_}, {p__:.2e}, {p1_:.2e}, {100*abs(p1_-p__)/p__:.2f}%, {10.**np.floor(np.log10(p__)):.2e}, {p2_:.2e},  {100*abs(p2_-p__)/p__:.2f}%, {p__:.2e}")

plt.show()
