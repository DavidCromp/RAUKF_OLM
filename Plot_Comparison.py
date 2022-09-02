from OLM import OLM_Single
from neuron import h
import numpy as np
from multicompartment.record import record_V_cai_ica_dend,\
    record_V_and_Cai, record_soma
from comp_opt_essential.record_1comp import set_up_full_recording
import matplotlib.pyplot as plt
from ukf import UkfEstimator
from tqdm import tqdm
from random import choice
from itertools import cycle
from pathos.multiprocessing import ProcessingPool as Pool
p = Pool(4)

multi=True;

# 
# Generate figure comparing SINGLE, FULL, and RAUKF
# For a given Step current. Refer to function definitions
# in main.py
#

if multi:
    h.load_file("./multicompartment/init_model_tweaked.hoc")
else:
    h.load_file("./comp_opt_essential/init_1comp.hoc")
    h.soma.ena=90
    h.soma.gna_Nasoma = 48.472e-4
    h.soma.gbar_Ikdrf = 43.343e-4
    h.soma.gbar_IM    = 0.19082e-4
    h.soma.gbar_Ika   = 73.062e-4

h.cvode.active(False)

# gf = h.Gfluct2(h.soma(0.5))

dt = h.dt

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

def figStep(I_Step):
    ts,I  = genStim([I_Step],nstim=1,time_scale=1)

    rec=None
    if multi:
        rec    = record_soma(h,0)
    else:
        rec    = set_up_full_recording(h,False)
    # spikes = set_up_spike_count(h)

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

    obs = np.array([np.array(rec[0])])[:,:].T
    obs += np.random.normal(0,3**0.5,size=obs.shape)


    theta_keys = ["g_M",
                  "g_kdrf",
                  "g_ka",
                  "g_nasoma"
                  ]

    nrn = OLM_Single()

    # Poor
    p1 = np.array([
        1.66e-05,
        5.53e-03,
        7.15e-03,
        4.56e-03
    ])
    # Optimal
    p2 = np.array([
        1.84e-05,
        5.25e-03,
        7.17e-03,
        4.55e-03
    ])
    p_ = np.array([nrn.p[key] for key in theta_keys])

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

    x_1, x_2, x__ = p.map(runPyWithP,[p1,p2,p_])

    x_1 = x_1[:,0]
    x_2 = x_2[:,0]
    x__ = x__[:,0]

    ## Figure Generation

    saveFig=True
    filetype='.svg'

    plt.rcParams['font.size'] = 8
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['lines.linewidth'] = 0.75
    plt.rcParams['legend.loc'] = 'upper right'
    figsize = (18/2.54,18/2.54/((1+5**0.5)/2))
    figsqar = (18/2.54,18/2.54)

    cs = [
        "#F21818",
        "#15A743",
        "#0952B1",
        "#ECE646",
        "#d0d"
    ]

    full="_Full" if multi else ""
    model="FULL" if multi else "SINGLE"

    OFFSET=len(obs)
    plt.figure(figsize=figsize)
    plt.subplot(221)
    plt.plot(ts,np.array(rec[0])[:OFFSET],color='k',label=f'{model}')
    plt.plot(ts,x_1,color=cs[0],label='Optimal Initial')
    plt.plot(ts,x_2,color=cs[1],label='Poor Initial')
    if multi:
        plt.plot(ts,x__,color=cs[2],label='SINGLE')
    xl = plt.xlim()
    yl_ = plt.ylim()
    yl = (yl_[0],yl_[1] + 10)
    plt.ylim(yl)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    plt.ylabel("Membrane Potential (mV)")
    plt.legend()
    plt.tight_layout()

    plt.subplot(222)
    plt.plot(ts,x_1,color=cs[0],label='Optimal Initial')
    plt.xlim(xl)
    plt.ylim(yl)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=False,      # ticks along the bottom edge are off
        right=False,         # ticks along the top edge are off
        labelleft=False) # labels along the bottom edge are off
    plt.legend()
    plt.tight_layout()

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
        plt.legend()
        plt.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            right=False,         # ticks along the top edge are off
            labelleft=False) # labels along the bottom edge are off

        plt.tight_layout()

    if saveFig:
        plt.savefig(f"img/KF{full}_Perf_Subplot_{len(theta_keys)}_param_{I_Step}{filetype}")

    plt.show()

_=list(map(figStep,[30,60,90]))
p.close()

