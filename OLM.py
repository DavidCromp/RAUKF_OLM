import numpy as np
from tqdm import tqdm
from numpy import exp
from scipy.optimize import fsolve

def vtrap(x,y):
    return np.where(np.abs(x/y)<1e-6, y*(1-x/y/2), x/(exp(x/y)-1))

class OLM_Single:
    """Single Compartment OLM Cell Model
    Derived from reduced single compartment OLM model written in NEURON"""

    def __init__(self):
        # Dynamic States
        v = -74
        self.vshift = -4.830346371483079

        # Parameters
        self.p = {
            "g_passsd": 7.5833e-06,
            "E_passsd": -64.64,
            "g_M"     : 0.19082e-4,#0.16916079274301054*1e-4,#1.6916e-5,
            "E_M"     : -95,
            "g_kdrf"  : 43.343e-4,#127.25184659083075*1e-4,#0.012725,
            "E_kdrf"  : -95,
            "g_ka"    : 73.062e-4,#70.16860025297115*1e-4,#0.0074107,
            "E_ka"    : -95,
            "g_nasoma": 48.472e-4,#74.10667249586207*1e-4,#0.0074107,
            "E_nasoma": 90,
            "g_r"     : 1.06309e-5,#1.0631e-5,
            "E_r"     : -34.0056,
            "cm"      : 0.27008*1e-3, # uF -> mF,
            "c1"      : 0.003,
            "c2"      : 0.003,
            "k1"      : 15,
            "k2"      : 15,
            "vhalf1"  : -63,
            "vhalf2"  : -63,
            "vhalfm"  : -33,
            "gmm"     : 0.7,
            "a0m"     : 0.036,
            "zetam"   : 0.1,
            "a0h"     : 0.17,
            "vhalfh"  : -105,
            "f"       : 0.92,
            "v_half"  : -103.69,
            "k"       : 9.9995804,
            "t1"      : 8.5657797,#8.5658,
            "t2 "     : 0.0296317,#0.029632,
            "t3"      : -6.9145,
            "t4"      : 0.1803,
            "t5"      : 4.3566601e-05,#4.3567e-5,
            "vshift"  : -4.830346371483079,#-4.8303,
            "I_bias":0
        }
        self.P = np.array([[v for v in self.p.values()]])

        self.x = np.array([[
            v,                   # V
            self.minf_M(v),      # m_M
            self.minf_kdrf(v),   # m_kdrf
            self.hinf_kdrf(v,self.P)[0],   # h_kdrf
            self.minf_ka(v),     # m_ka
            self.hinf_ka(v),     # h_ka
            self.minf_nasoma(v), # m_nasoma
            self.hinf_nasoma(v), # h_nasoma
            0                    # r
        ]])
        self.dt = 0.1
        self.SA = 31403.36016528357 * 1e-8 # um^2 -> cm^2
        self.qt = 3**((34-23)/10)
    def test(self,tstop,I):
        dt = self.dt
        P  = self.P
        ks=int(tstop/dt)+1
        xs = np.zeros((ks,self.x.shape[1]))
        for k in tqdm(range(1,ks)):
            self.x = self.forward(self.x,I,P)
            xs[[k]] = self.x
        return xs

    def rk4(self,x,p,I):
        dt = self.dt
        k1 = dt * self.step(x, p, I)
        k2 = dt * self.step(x + k1 / 2.0, p, I)
        k3 = dt * self.step(x + k2 / 2.0, p, I)
        k4 = dt * self.step(x + k3, p, I)
        return x + k1 / 6.0 + k2 / 3.0 + k3 / 3.0 + k4 / 6.0
    def forward(self, x_k_1, I, p, int_factor=1):
        return self.step_be(x_k_1,p,I).T

    def mtau_M(self,v,p):
        c1 = p[:,13]
        c2 = p[:,14]
        k1 = p[:,15]
        k2 = p[:,16]
        vhalf1 = p[:,17]
        vhalf2 = p[:,18]

        a = c1/exp(-(v-vhalf1)/k1)
        b = c2/exp((v-vhalf2)/k2)
        mtau = 1/(a+b)
        # Numerical Stability
        mmin = 7
        return np.where(mtau<mmin,mmin,mtau)
    def mtau_kdrf(self,v,p):
        vhalfm   = p[:,19]
        gmm      = p[:,20]
        qt       = self.qt
        a0m      = p[:,21]

        zetam = p[:,22]
        alpm  = exp(zetam*(v-vhalfm))
        betm  = exp(zetam*gmm*(v-vhalfm))
        mtau  = betm/(qt*a0m*(1+alpm))
        return mtau
    def mtau_ka(self,v):
        qt   = self.qt
        mtau = 0.5/qt
        return mtau
    def mtau_nasoma(self,v,p):
        vshift = p[:,33]
        alpha = 0.1*vtrap(-(v+38-vshift),10)
        beta = 4*exp(-(v+63-vshift)/18)

        mtau = 1/(alpha + beta)
        return mtau

    def htau_kdrf(self,v):
        return 1000
    def htau_ka(self,v,p):
        a0h      = p[:,23]
        vhalfh   = p[:,24]
        qt       = self.qt

        htau = a0h*(v-vhalfh)/qt
        hmin  = 5
        return np.where(htau<hmin/qt,hmin/qt,htau)
    def htau_nasoma(self,v):
        vshift = self.vshift

        alpha = 0.07*exp(-(v+63-vshift)/20)
        beta = 1/(1+exp(-(v+33-vshift)/10))
        htau = 1/(alpha + beta)
        return htau

    def minf_M(self,v):
        return 1/(1 + exp(-(v+27)/7))
    def minf_kdrf(self,v):
        return (1/(1 + exp(-(v+36.2)/16.1)))**4
    def minf_ka(self,v):
        return (1/(1 + exp(-(v+41.4)/26.6)))**4
    def minf_nasoma(self,v):
        alpha = 0.1*vtrap(-(v+38-self.vshift),10)
        beta = 4*exp(-(v+63-self.vshift)/18)
        mtau_nasoma = 1/(alpha + beta)
        return alpha*mtau_nasoma

    def hinf_kdrf(self,v,p):
        f = p[:,25]
        return f*(1/(1 + np.exp((v+40.6)/7.8)))+(1-f)
    def hinf_ka(self,v):
        return 1/(1 + exp((v+78.5)/6))
    def hinf_nasoma(self,v):
        alpha = 0.07*exp(-(v+63-self.vshift)/20)
        beta = 1/(1+exp(-(v+33-self.vshift)/10))
        htau_nasoma = 1/(alpha + beta)
        return alpha*htau_nasoma

    def rinf(self,v,p):
        v_half   = p[:,26]
        k        = p[:,27]

        return 1/(1 + exp((v-v_half)/k))

    def rtau(self,v,p):
        t1       = p[:,28]
        t2       = p[:,29]
        t3       = p[:,30]
        t4       = p[:,31]
        t5       = p[:,32]

        return 1/(exp(-t1-t2*v) + exp(-t3+t4*v)) + t5

    def step(self,x,p,I):
        # Parameter localization
        # Dynamic States
        v      = x[:,0]

        m_M    = x[:,1]

        m_kdrf = x[:,2]
        h_kdrf = x[:,3]

        m_ka   = x[:,4]
        h_ka   = x[:,5]

        m_nasoma = x[:,6]
        h_nasoma = x[:,7]

        r        = x[:,8]
        # Static States
        g_passsd = p[:,0]
        E_passsd = p[:,1]

        g_M      = p[:,2]
        E_M      = p[:,3]

        g_kdrf   = p[:,4]
        E_kdrf   = p[:,5]

        g_ka     = p[:,6]
        E_ka     = p[:,7]

        g_nasoma = p[:,8]
        E_nasoma = p[:,9]

        g_r      = p[:,10]
        E_r      = p[:,11]

        cm       = p[:,12]

        I_bias   = p[:,34]

        # State Propogation
        # passsd
        I_passsd = g_passsd*(v-E_passsd)
        # IM
        I_M = g_M*m_M*(v-E_M)

        minf_M = self.minf_M(v)
        mtau_M = self.mtau_M(v,p)
        m_M_ = (minf_M-m_M)/mtau_M

        # Ikdrf
        I_kdrf = g_kdrf * m_kdrf * h_kdrf * (v - E_kdrf)

        minf_kdrf = self.minf_kdrf(v)
        mtau_kdrf = self.mtau_kdrf(v,p)
        m_kdrf_ = (minf_kdrf - m_kdrf)/mtau_kdrf

        hinf_kdrf = self.hinf_kdrf(v,p)
        htau_kdrf = self.htau_kdrf(v)
        h_kdrf_ = (hinf_kdrf - h_kdrf)/htau_kdrf
        # Ika
        I_ka = g_ka * m_ka * h_ka * (v-E_ka)
        minf_ka = self.minf_ka(v)
        mtau_ka = self.mtau_ka(v)
        m_ka_ = (minf_ka-m_ka)/mtau_ka

        hinf_ka = self.hinf_ka(v)
        htau_ka = self.htau_ka(v,p)
        h_ka_ = (hinf_ka-h_ka)/htau_ka

        # Nasoma
        I_nasoma = g_nasoma * m_nasoma*m_nasoma*m_nasoma * h_nasoma * (v-E_nasoma)

        mtau_nasoma = self.mtau_nasoma(v,p)
        minf_nasoma = self.minf_nasoma(v)

        htau_nasoma = self.htau_nasoma(v)
        hinf_nasoma = self.hinf_nasoma(v)

        m_nasoma_ = (minf_nasoma - m_nasoma)/mtau_nasoma
        h_nasoma_ = (hinf_nasoma - h_nasoma)/htau_nasoma
        # Ih
        I_h = g_r * r * (v-E_r)

        rinf = self.rinf(v,p)
        tau_r = self.rtau(v,p)
        r_ = (rinf - r) / tau_r

        v_ = (-(I_passsd+I_M+I_kdrf+I_ka+I_nasoma+I_h)*(self.SA)
              + I + I_bias) / (cm*self.SA)

        return np.array([
            v_,
            m_M_,
            m_kdrf_,
            h_kdrf_,
            m_ka_,
            h_ka_,
            m_nasoma_,
            h_nasoma_,
            r_,
        ])
    def step_be(self,x,p,I):
        # Parameter localization
        dt     = self.dt
        # Dynamic States
        v      = x[:,0]

        m_M    = x[:,1]

        m_kdrf = x[:,2]
        h_kdrf = x[:,3]

        m_ka   = x[:,4]
        h_ka   = x[:,5]

        m_nasoma = x[:,6]
        h_nasoma = x[:,7]

        r        = x[:,8]
        # Static States
        g_passsd = p[:,0]
        E_passsd = p[:,1]

        g_M      = p[:,2]
        E_M      = p[:,3]

        g_kdrf   = p[:,4]
        E_kdrf   = p[:,5]

        g_ka     = p[:,6]
        E_ka     = p[:,7]

        g_nasoma = p[:,8]
        E_nasoma = p[:,9]

        g_r      = p[:,10]
        E_r      = p[:,11]

        cm       = p[:,12]

        I_bias   = p[:,34]

        # State Propogation
        ## Voltage evaluated  at t+dt
        ## Currents evaluated at t+dt
        ## States evalutated  at t+dt/2
        # passsd
        # IM

        minf_M = self.minf_M(v)
        mtau_M = self.mtau_M(v,p)
        m_M_ = m_M + (1 - exp(-(dt / mtau_M)))*(minf_M - m_M)

        # Ikdrf

        minf_kdrf = self.minf_kdrf(v)
        mtau_kdrf = self.mtau_kdrf(v,p)
        m_kdrf_ = m_kdrf + (1 - exp(-(dt / mtau_kdrf)))*(minf_kdrf - m_kdrf)

        hinf_kdrf = self.hinf_kdrf(v,p)
        htau_kdrf = self.htau_kdrf(v)
        h_kdrf_ = h_kdrf + (1 - exp(-(dt / htau_kdrf)))*(hinf_kdrf - h_kdrf)
        # Ika
        minf_ka = self.minf_ka(v)
        mtau_ka = self.mtau_ka(v)
        m_ka_ = m_ka + (1 - exp(-(dt / mtau_ka)))*(minf_ka - m_ka)

        hinf_ka = self.hinf_ka(v)
        htau_ka = self.htau_ka(v,p)
        h_ka_ = h_ka + (1 - exp(-(dt / htau_ka)))*(hinf_ka - h_ka)

        # Nasoma
        mtau_nasoma = self.mtau_nasoma(v,p)
        minf_nasoma = self.minf_nasoma(v)

        htau_nasoma = self.htau_nasoma(v)
        hinf_nasoma = self.hinf_nasoma(v)

        m_nasoma_ = m_nasoma+(1-exp(-(dt/mtau_nasoma)))*(minf_nasoma-m_nasoma)
        h_nasoma_ = h_nasoma+(1-exp(-(dt/htau_nasoma)))*(hinf_nasoma-h_nasoma)
        # Ih
        rinf = self.rinf(v,p)
        rtau = self.rtau(v,p)
        r_ = r + (1 - exp(-(dt / rtau)))*(rinf - r)

        I_passsd = lambda x: g_passsd*(x-E_passsd)
        I_M = lambda x: g_M * m_M_ * (x - E_M)
        I_kdrf = lambda x: g_kdrf * m_kdrf_ * h_kdrf_ * (x - E_kdrf)
        I_ka = lambda x: g_ka * m_ka_ * h_ka_ * (x-E_ka)
        I_nasoma = lambda x: g_nasoma*m_nasoma_**3*h_nasoma_*(x-E_nasoma)
        I_h = lambda x: g_r * r_ * (x-E_r)

        v_ = fsolve(
            lambda x: v + dt*(
                -(I_passsd(x)+
                  I_M(x)+
                  I_kdrf(x)+
                  I_ka(x)+
                  I_nasoma(x)+
                  I_h(x))*(self.SA)
                + I + I_bias) / (cm*self.SA) - x,v,
            xtol=1e-9
        )

        return np.array([
            v_,
            m_M_,
            m_kdrf_,
            h_kdrf_,
            m_ka_,
            h_ka_,
            m_nasoma_,
            h_nasoma_,
            r_,
        ])
    def observe(self,x):
        return x[:, [0]]
    def getCurrents(self):
        x = self.x
        p = self.P
        # Dynamic States
        v      = x[:,0]
        m_M    = x[:,1]
        m_kdrf = x[:,2]
        h_kdrf = x[:,3]
        m_ka   = x[:,4]
        h_ka   = x[:,5]
        m_nasoma = x[:,6]
        h_nasoma = x[:,7]
        r        = x[:,8]
        # Static States
        g_passsd = p[:,0]
        E_passsd = p[:,1]

        g_M      = p[:,2]
        E_M      = p[:,3]

        g_kdrf   = p[:,4]
        E_kdrf   = p[:,5]

        g_ka     = p[:,6]
        E_ka     = p[:,7]

        g_nasoma = p[:,8]
        E_nasoma = p[:,9]

        g_r      = p[:,10]
        E_r      = p[:,11]

        I_passsd = g_passsd*(v-E_passsd)
        I_M = g_M*m_M*(v-E_M)
        I_kdrf = g_kdrf * m_kdrf * h_kdrf * (v - E_kdrf)
        I_ka = g_ka * m_ka * h_ka * (v-E_ka)
        I_nasoma = g_nasoma * m_nasoma*m_nasoma*m_nasoma * h_nasoma * (v-E_nasoma)
        I_h = g_r * r * (v-E_r)

        return np.array([I_ka,I_kdrf,I_M,I_passsd,I_nasoma,I_h])
