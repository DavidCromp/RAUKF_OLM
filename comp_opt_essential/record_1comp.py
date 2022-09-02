def set_up_full_recording(h,OU=True):
    l=0
    v_vecD4 = h.Vector()
    v_vecD4.record(h.soma(l)._ref_v)

    IhD4 = h.Vector()
    IhD4.record(h.soma(l)._ref_ih_Ih)

    INaD4 = h.Vector()
    INaD4.record(h.soma(l)._ref_ina_Nasoma)

    IKaD4 = h.Vector()
    IKaD4.record(h.soma(l)._ref_ik_Ika)

    IKdrfD4= h.Vector()
    IKdrfD4.record(h.soma(l)._ref_ik_Ikdrf)

    ImD4 = h.Vector()
    ImD4.record(h.soma(l)._ref_ik_IM)

    IlD4 = h.Vector()
    IlD4.record(h.soma(l)._ref_i_passsd)

    Ig = 0 if not OU else h.Vector().record(h.Gfluct2[0]._ref_i)
    
    I  = h.Vector().record(h.soma(l)._ref_i_cap)

    recording = [v_vecD4, IKaD4, IKdrfD4, ImD4, IlD4, INaD4, IhD4, Ig,I]

    return recording


def set_up_spike_count(h):
    l=0.5
    apc = h.APCount(h.soma(l))
    apc.thresh = -20
    spike_timing = h.Vector()
    apc.record(spike_timing)
    return spike_timing
