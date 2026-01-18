
"""
"""

from copy import copy
import numpy as np
import pymc
import wfpt
from wfpt_addm import wiener_like_addm

print('wfpt_addm')

from kabuki.hierarchical import Knode
from kabuki.utils import stochastic_from_dist
from hddm.models import HDDM

class HDDM_addm(HDDM):
    """adaptive DDM model
    """

    def __init__(self, *args, **kwargs):
        self.non_centered = kwargs.pop('non_centered', False)
        self.alpha = kwargs.pop('alpha', True)
        self.wfpt_addm_class = WienerADDM

        super(HDDM_addm, self).__init__(*args, **kwargs)
        
    def _create_stochastic_knodes(self, include):
        knodes = super(HDDM_addm, self)._create_stochastic_knodes(include)
        std_upper_normal=10   
        std_group=10
        if self.alpha:
            knodes.update(self._create_family_normal(
                'v', value=0, g_mu=0, g_tau=std_group**-2, std_lower=1e-10, std_upper=std_upper_normal, std_value=.1))                   
            knodes.update(self._create_family_normal(
                'alpha', value=0, g_mu=0, g_tau=std_group**-2, std_lower=1e-10, std_upper=std_upper_normal, std_value=.1))    
            knodes.update(self._create_family_normal(
                'om', value=0, g_mu=0, g_tau=(std_group/20)**-2, std_lower=1e-10, std_upper=std_upper_normal/10, std_value=.1))
        return knodes

    def _create_wfpt_parents_dict(self, knodes):
        wfpt_parents = super(HDDM_addm, self)._create_wfpt_parents_dict(knodes)
        wfpt_parents['v'] = knodes['v_bottom']        
        wfpt_parents['alpha'] = knodes['alpha_bottom']     
        wfpt_parents['om'] = knodes['om_bottom']        
        return wfpt_parents

    def _create_wfpt_knode(self, knodes):
        wfpt_parents = self._create_wfpt_parents_dict(knodes)
        return Knode(self.wfpt_addm_class, 'wfpt', observed=True, col_name=['subj_idx', 'rt', 'response', 'conflict', 'starting_trial'], **wfpt_parents)

def wienerADDM_like(x, alpha, om, v, sv, a, z, sz, t, st, p_outlier=0):

    wiener_params = {'err': 1e-4, 'n_st': 2, 'n_sz': 2,
                     'use_adaptive': 1,
                     'simps_err': 1e-3,
                     'w_outlier': 0.1}
    wp = wiener_params    
    return wiener_like_addm(x['rt'].values, x['conflict'].values, x['starting_trial'].values, alpha, om, v, sv, a, z, sz, t, st, p_outlier=p_outlier, **wp)
WienerADDM = stochastic_from_dist('wienerADDM', wienerADDM_like)