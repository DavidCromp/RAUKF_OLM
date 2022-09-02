from neuron import h, gui
import numpy

"""This functino sets up recording for soma and D4 location and return them in a tuple of list of vectors"""


def set_up_full_recording(h,dendnum,OU=False):

    v_vecD4 = h.Vector()
    v_vecD4.record(h.dend[dendnum](0)._ref_v)

    IhD4 = h.Vector()
    IhD4.record(h.dend[dendnum](0)._ref_ih_Ih)

    INaD4 = h.Vector()
    INaD4.record(h.dend[dendnum](0)._ref_ina_Nadend)

    IKaD4 = h.Vector()
    IKaD4.record(h.dend[dendnum](0)._ref_ik_Ika)

    IKdrfD4= h.Vector()
    IKdrfD4.record(h.dend[dendnum](0)._ref_ik_Ikdrf)

    IKdrsD4 = h.Vector()
    IKdrsD4.record(h.dend[dendnum](0)._ref_ik_Ikdrs)

    ImD4 = h.Vector()
    ImD4.record(h.dend[dendnum](0)._ref_ik_IM)

    IKCaD4 = h.Vector()
    IKCaD4.record(h.dend[dendnum](0)._ref_ik_kca)

    ICaLD4 = h.Vector()
    ICaLD4.record(h.dend[dendnum](0)._ref_ica_cal)

    ICaTD4 = h.Vector()
    ICaTD4.record(h.dend[dendnum](0)._ref_ica_cat)

    IlD4 = h.Vector()
    IlD4.record(h.dend[dendnum](0)._ref_i_passsd)

    Ige = 0 if not OU else h.Vector().record(h.Gfluct2[0]._ref_g_e)
    Igi = 0 if not OU else h.Vector().record(h.Gfluct2[0]._ref_g_i)
    Ig = 0 if not OU else h.Vector().record(h.Gfluct2[0]._ref_i)
    
    recording = [v_vecD4, IKaD4, IKdrfD4, IKdrsD4, IKCaD4, ImD4, IlD4, INaD4, ICaTD4, ICaLD4, IhD4, Ige,Igi,Ig]

    return recording


def record_soma(h,somanum):
    v_vecD4 = h.Vector()
    v_vecD4.record(h.soma[somanum](0.5)._ref_v)

    IhD4 = h.Vector()
    IhD4.record(h.soma[somanum](0.5)._ref_ih_Ih)

    INaD4 = h.Vector()
    INaD4.record(h.soma[somanum](0.5)._ref_ina_Nasoma)

    IKaD4 = h.Vector()
    IKaD4.record(h.soma[somanum](0.5)._ref_ik_Ika)

    IKdrfD4 = h.Vector()
    IKdrfD4.record(h.soma[somanum](0.5)._ref_ik_Ikdrf)

    IKdrsD4 = h.Vector()
    IKdrsD4.record(h.soma[somanum](0.5)._ref_ik_Ikdrs)

    ImD4 = h.Vector()
    ImD4.record(h.soma[somanum](0.5)._ref_ik_IM)

    IlD4 = h.Vector()
    IlD4.record(h.soma[somanum](0.5)._ref_i_passsd)

    recording = [v_vecD4, IKaD4, IKdrfD4, IKdrsD4, ImD4, IlD4, INaD4, IhD4]

    return recording


def record_V_and_Cai(h):
    v_vec = h.Vector()
    v_vec.record(h.dend[0](0)._ref_v)
    Cai = h.Vector()
    Cai.record(h.dend[0](0)._ref_cai)

    d4_comp = 50

    v_vecd4 = h.Vector()
    v_vecd4.record(h.dend[d4_comp](0)._ref_v)
    Caid4 = h.Vector()
    Caid4.record(h.dend[d4_comp](0)._ref_cai)

    return [v_vec, Cai], [v_vecd4, Caid4]


def record_ica_and_Cai(h):
    ica = h.Vector()
    ica.record(h.dend[0](0)._ref_ica)
    Cai = h.Vector()
    Cai.record(h.dend[0](0)._ref_cai)

    d4_comp = 50

    icad4 = h.Vector()
    icad4.record(h.dend[d4_comp](0)._ref_ica)
    Caid4 = h.Vector()
    Caid4.record(h.dend[d4_comp](0)._ref_cai)

    return [ica, Cai], [icad4, Caid4]


def record_V_cai_ica_dend(h,dend_num):
    ica = h.Vector()
    ica.record(h.dend[dend_num](0)._ref_ica)
    v = h.Vector()
    v.record(h.dend[dend_num](0)._ref_v)
    cai = h.Vector()
    cai.record(h.dend[dend_num](0)._ref_cai)

    return v, cai, ica
