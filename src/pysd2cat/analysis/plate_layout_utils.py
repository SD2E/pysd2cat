import pandas as pd
import logging


l = logging.getLogger(__file__)
l.setLevel(logging.INFO)



def get_samples_from_condition_set(factors, condition_set, parameters = None):
    """
    Compute factorial experiment from condition_set and parameters.
    """

    if not condition_set_is_singletons(factors, condition_set):
        samples = condition_set_cross_product(factors, condition_set)
    else:
        samples = pd.DataFrame({ factor['factor'] : factor['values'] for factor in condition_set['factors']})


    if 'key' in samples.columns:
        samples = samples.drop(columns=['key'])

    if parameters:
        for parameter, value in parameters.items():
            if len(samples) == 0:
                samples[parameter] = [str(value)]
            else:
                samples.loc[:, parameter] = str(value)

    # Add columns for factors not present in the condtiion_set
    for factor in factors:
        if factor not in samples.columns:
            samples.loc[:, factor] = "TBD"

    
    return samples

def get_factor_from_condition_set(factor_id, condition_set):
    for cs_factor in condition_set['factors']:
        if cs_factor['factor'] == factor_id:
            return cs_factor
    raise Exception("Could not find factor %s in condition_set %s", factor_id, condition_set)

def condition_set_is_singletons(factors, condition_set):
    for factor_id in factors:
        factor = get_factor_from_condition_set(factor_id, condition_set)
        if len(factor['values']) != 1:
            return False
    return True

def condition_set_cross_product(factors, condition_set):
    samples = pd.DataFrame()
    for factor_id in factors:
        factor = get_factor_from_condition_set(factor_id, condition_set)
        #if factors[factor['factor']]['ftype'] == 'time':
        #    continue

#        if "" in factor['values']:
#            factor['values'].remove("")
        
        l.debug("Merging factor %s = %s", factor['factor'], factor['values'])

        if len(factor['values']) == 0:
            l.debug("Skipping factor %s, no values", factor['factor'])
            continue
        
        if len(samples) == 0:
            samples = pd.DataFrame({factor['factor'] : factor['values']})
        else:
            samples.loc[:,'key'] = 0
            fdf = pd.DataFrame({factor['factor'] : factor['values']})
            #l.info(fdf)
            fdf.loc[:,'key'] = 0
            samples = samples.merge(fdf, how='left', on='key')
    return samples
