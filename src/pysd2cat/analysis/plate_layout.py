from pysmt.shortcuts import Symbol, And, Or, Not, Implies, Equals, Iff, is_sat, get_model, GT, GE, LT, Int, String, TRUE
from pysmt.typing import INT, StringType


import pandas as pd

import logging


l = logging.getLogger(__file__)
l.setLevel(logging.DEBUG)



def generate_variables1(inputs):
    """
    Encoding variables and values
    """

    
    samples = inputs['samples']
    factors = inputs['factors']
    containers = inputs['containers']
    aliquots = [a for c in containers for a in containers[c]['aliquots']]

    variables = {}
    variables['reverse_index'] = {}
    #print(samples)
#    variables['tau_symbols'] = \
#      {
#          a : {
#              x : Symbol("tau_{}".format(a[x]))
#              for x in samples[a] 
#            }
#            for a in samples
#      }
#    for aliquot in variables['tau_symbols']:
#        for sample in variables['tau_symbols'][aliquot]:
#            var = variables['tau_symbols'][aliquot][sample]
#            if var not in variables['reverse_index']:                
#                variables['reverse_index'][var] = {}
#            variables['reverse_index'][var].update({"sample": "x{}_{}".format(sample, aliquot), "aliquot" : aliquot})
            
    variables['aliquot_factors'] = \
      {
          c: {
              a : {
                  factor_id : {
                      level : Symbol("{}({}_{})={}".format(factor_id, a, c, level))
                      for level in factor['domain']
                  }
                  for factor_id, factor in factors.items() if factor['ftype'] == "aliquot"                
              }
              for a in samples[c]
          }
          for c in samples
      }
    
    variables['sample_factors'] = \
      {
          c : {
              a : {
                  sample: {
                      factor_id : {
                          level : Symbol("{}(x{}_{}_{})={}".format(factor_id, sample, a, c, level))
                          for level in factor['domain']
                      }
                      for factor_id, factor in factors.items() if factor['ftype'] == "sample"
                  }
                  for sample in samples[c][a]
              }
              for a in samples[c]
          }
          for c in samples
      }
    for container in variables['sample_factors']:
        for aliquot in variables['sample_factors'][container]: 
            for sample in variables['sample_factors'][container][aliquot]:
                for factor_id in variables['sample_factors'][container][aliquot][sample]:
                    for level in variables['sample_factors'][container][aliquot][sample][factor_id]:
                        var = variables['sample_factors'][container][aliquot][sample][factor_id][level]
                        if var not in variables['reverse_index']:                
                            variables['reverse_index'][var] = {}
                        variables['reverse_index'][var].update({"type" : "sample", "aliquot" : aliquot, "sample" :  "x{}_{}_{}".format(sample, aliquot, container), "container" : container, factor_id : level})
            for factor_id in variables['aliquot_factors'][container][aliquot]:
                for level in variables['aliquot_factors'][container][aliquot][factor_id]:
                    var = variables['aliquot_factors'][container][aliquot][factor_id][level]
                    if var not in variables['reverse_index']:                
                        variables['reverse_index'][var] = {}
                    variables['reverse_index'][var].update({"type" : "aliquot", "aliquot" : aliquot, "container" : container, factor_id : level})

        
    values = {}

    variables['exp_factor'] = \
      {
          factor_id : {
              level : Symbol("{}_exp={}".format(factor_id, level))
              for level in factor['domain'] 
              }
            for factor_id, factor in factors.items() if factor['ftype'] == "experiment"
      }
    for exp_factor in variables['exp_factor']:
        for level in variables['exp_factor'][exp_factor]:
            var = variables['exp_factor'][exp_factor][level]
            if var not in variables['reverse_index']:
                variables['reverse_index'][var] = {}
            variables['reverse_index'][var].update({"type" : "experiment", exp_factor : level})
            
    variables['batch_factor'] = \
      {
          factor_id : {
                level : {
                    container : Symbol("{}_{}_batch={}".format(factor_id, container, level))
                    for container in containers
                    }
                  for level in factor['domain']   
              }
              for factor_id, factor in factors.items() if factor['ftype'] == "batch"
      }
    for batch_factor in variables['batch_factor']:
        for level in variables['batch_factor'][batch_factor]:
            for container in variables['batch_factor'][batch_factor][level]:
                var = variables['batch_factor'][batch_factor][level][container]
                if var not in variables['reverse_index']:
                    variables['reverse_index'][var] = {}
                variables['reverse_index'][var].update({"type" : "batch", "container" : container, batch_factor : level})

    variables['column_factor'] = \
      {
          factor_id : {
              level : {
                  col : Symbol("{}_{}_col={}".format(factor_id, col, level))  
                  for _, container in containers.items()
                  for col in container['columns']
                  }
                  for level in factor['domain']   
            }
          for factor_id, factor in factors.items() if factor['ftype'] == "column"
      }
    for column_factor in variables['column_factor']:
        for level in variables['column_factor'][column_factor]:
            for column in variables['column_factor'][column_factor][level]:
                var = variables['column_factor'][column_factor][level][column]
                if var not in variables['reverse_index']:
                    variables['reverse_index'][var] = {}
                container = [ c for c in containers if column in containers[c]['columns']][0]
                variables['reverse_index'][var].update({"type" : "column", "column" : column, "container" : container, column_factor : level})

    l.debug("Variables: %s", variables)
    l.debug("Values: %s", values)
    return variables, values

def generate_variables(inputs):
    """
    Encoding variables and values
    """

    
    samples = inputs['samples']
    factors = inputs['factors']
    containers = inputs['containers']
    aliquots = [a for c in containers for a in containers[c]['aliquots']]

    variables = {}
    variables['reverse_index'] = {}
    variables['tau_symbols'] = \
      {
          a : {
              x:  Symbol("tau_{}".format(x, a))             
              for x in samples
            }
            for a in aliquots
      }
    for aliquot in variables['tau_symbols']:
        for sample in variables['tau_symbols'][aliquot]:
            var = variables['tau_symbols'][aliquot][sample]
            if var not in variables['reverse_index']:                
                variables['reverse_index'][var] = {}
            variables['reverse_index'][var].update({"sample": sample, "aliquot" : aliquot})
            
    variables['tau_symbols_perp'] = { x: Symbol("tau_{}=perp".format(x)) for x in samples }
    variables['sample_factors'] = \
      { 
        sample: {
            factor_id : {
                level : Symbol("{}({})={}".format(factor_id, sample, level))
                for level in factor['domain']
                }
                for factor_id, factor in factors.items()
        }
        for sample in samples
      }
    for sample in variables['sample_factors']:
        for factor_id in variables['sample_factors'][sample]:
            for level in variables['sample_factors'][sample][factor_id]:
                var = variables['sample_factors'][sample][factor_id][level]
                if var not in variables['reverse_index']:                
                    variables['reverse_index'][var] = {}
                variables['reverse_index'][var].update({"sample" : sample, factor_id : level})

    
    variables['sample_factors_perp'] = \
      { 
          sample: {
              factor_id : Symbol("{}({})=perp".format(factor_id, sample))
              for factor_id, factor in factors.items()
            }  for sample in samples
      }
    
    values = {}
    values['perp'] = Int(-1)
    values['min_aliquot'] = Int(0)
    values['max_aliquot'] = Int(len(aliquots))

    variables['exp_factor'] = \
      {
          factor_id : {
              level : Symbol("{}_exp={}".format(factor_id, level))
              for level in factor['domain'] 
              }
            for factor_id, factor in factors.items() if factor['ftype'] == "experiment"
      }
    variables['batch_factor'] = \
      {
          factor_id : {
                level : {
                    container : Symbol("{}_{}_batch={}".format(factor_id, container, level))
                    for container in containers
                    }
                  for level in factor['domain']   
              }
              for factor_id, factor in factors.items() if factor['ftype'] == "batch"
      }
    variables['column_factor'] = \
      {
          factor_id : {
              level : {
                  col : Symbol("{}_{}_col={}".format(factor_id, col, level))  
                  for _, container in containers.items()
                  for col in container['columns']
                  }
                  for level in factor['domain']   
            }
          for factor_id, factor in factors.items() if factor['ftype'] == "column"
      }

    l.debug("Variables: %s", variables)
    l.debug("Values: %s", values)
    return variables, values



def generate_bounds(inputs, variables, values):
    """
    Generate bounds on variables.
    """
    samples = inputs['samples']
    factors = inputs['factors']
    sample_factors = variables['sample_factors']
    
    ## A sample can be mapped to None (perp) or one of the aliquots
    tau_bounds = \
      And([
        And(GT(x, values['perp']),
            LT(x, values['max_aliquot']))
          for k, x in variables['tau_symbols'].items()
          ])
    
    ## Each factor assigment must select a level from the factor domain
    factor_bounds = \
      And([
          And(GE(sample_factors[sample][factor], Int(0)),
              LT(sample_factors[sample][factor], Int(len(factors[factor]['domain'])))) 
              for sample in samples
          for factor in factors
          ])

    return And(tau_bounds, factor_bounds)

def generate_constraints(inputs):
    """
    Generate constraints for plate layout encoding.
    """
    variables, values = generate_variables(inputs)

    constraints = []

    #bounds = generate_bounds(inputs, variables, values)
    #constraints.append(bounds)
    
    samples = inputs['samples']
    factors = inputs['factors']
    containers = inputs['containers']
    requirements = inputs['requirements']

    aliquots = [a for c in containers for a in containers[c]['aliquots']]


    tau_symbols = variables['tau_symbols']
    tau_symbols_perp = variables['tau_symbols_perp']
    sample_factors = variables['sample_factors']
    sample_factors_perp = variables['sample_factors_perp']
    exp_factor = variables['exp_factor']
    batch_factor = variables['batch_factor']
    column_factor = variables['column_factor']

    perp = values['perp']

    
    # (1), Each aliquot has a sample mapped to it
    aliquot_sample_constraint = \
      And([
        Or([tau_symbols[x][a]
            for x in tau_symbols ])
        for a in aliquots
        ])
        
    l.debug("aliquot_sample_constraint: %s", aliquot_sample_constraint)
    constraints.append(aliquot_sample_constraint)
    
    # (2)
    mapped_are_assigned_constraint = \
      And([
          Iff(tau_symbols_perp[x], 
              And([ sample_factors_perp[x][f] for f in factors
            ])) 
          for x in samples
          ])
    l.debug("mapped_are_assigned_constraint: %s", mapped_are_assigned_constraint)
    constraints.append(mapped_are_assigned_constraint)
    
    # (3)
    uniformly_assigned_factors_constraint = \
      And([
          Implies(sample_factors_perp[x][f], 
                  sample_factors_perp[x][fp])
            for f in factors 
            for fp in factors
            for x in samples
          ])
    l.debug("uniformly_assigned_factors_constraint: %s", uniformly_assigned_factors_constraint)
    constraints.append(uniformly_assigned_factors_constraint)
    
    # (4)
    requirements_constraint = \
    And([Implies(Not(tau_symbols_perp[x]),
        Or([
            And([
                Or([sample_factors[x][f['factor']][level]                           
                    for level in f['values']]) 
                for f in r["factors"]]) 
            for r in requirements]))
        for x in samples])
    l.debug("requirements_constraint: %s", requirements_constraint)
    constraints.append(requirements_constraint)

    # (5)
    aliquot_properties_constraint = \
      And([Implies(tau_symbols[x][aliquot],
                   And([sample_factors[x][factor][level] 
                        for factor, level in aliquot_properties.items()]))
               for x in samples
               for _, c in containers.items()
               for aliquot, aliquot_properties in c['aliquots'].items()
            ])
    l.debug("aliquot_properties_constraint: %s", aliquot_properties_constraint)
    constraints.append(aliquot_properties_constraint)

    # (6)
    experiment_factors_constraint = \
    And([Implies(Not(tau_symbols_perp[x]),
                 And([
                     Or([And(sample_factors[x][factor_id][level],
                             exp_factor[factor_id][level])
                        for level in factor['domain']
                        ])
                     for factor_id, factor in factors.items() if factor["ftype"] == "experiment"
                     ]))
         for x in samples])
    l.debug("experiment_factors_constraint: %s", experiment_factors_constraint)
    constraints.append(experiment_factors_constraint)

    # (7)
    batch_factors_constraint = \
      And([ 
        Implies(tau_symbols[x][aliquot],
                And([
                     Or([And(sample_factors[x][factor_id][level],
                             batch_factor[factor_id][level][container_id])
                        for level in factor['domain']
                        ])
                     for factor_id, factor in factors.items() if factor["ftype"] == "batch"
                     ]))
        for container_id, container in containers.items()
        for aliquot, aliquot_properties in container['aliquots'].items()
        for x in samples
        ])
    l.debug("batch_factors_constraint: %s", batch_factors_constraint)
    constraints.append(batch_factors_constraint)

    # (8)
    sample_factors_constraint = \
    And([
        Implies(Or([And(tau_symbols[x][a], tau_symbols[xp][a])
                    for a in aliquots]),
                And([
                    Or([And(sample_factors[x][factor_id][level],
                            sample_factors[xp][factor_id][level])
                        for level in factor['domain']
                        ])
                     for factor_id, factor in factors.items()  if factor["ftype"] == "sample"]))
        for xp in samples
        for x in samples])
    l.debug("sample_factors_constraint: %s", sample_factors_constraint)
    constraints.append(sample_factors_constraint)

    # (9)
    if len(column_factor) > 0:
        column_factors_constraint = \
        And([ 
            Implies(tau_symbols[x][a], 
                    And([
                        Or([And(sample_factors[x][factor_id][level],
                                column_factor[factor_id][level][column_id])
                            for level in factor['domain']
                            ])
                        for factor_id, factor in factors.items() if factor["ftype"] == "column"]))    
            for x in samples           
            for container_id, container in containers.items()
            for column_id, column in container['columns'].items()
            for a in column
            ])
        constraints.append(column_factors_constraint)
        l.debug("column_factors_constraint: %s", column_factors_constraint)



    def factor_cross_product(factors, cross_product):
        if len(factors) == 0:
            return cross_product
        else:
            result = []
            factor = factors.pop(0)
            for elt in cross_product:
                for value in factor['values']:
                    expansion = elt.copy()
                    expansion.update({factor['factor'] : value})
                    result.append(expansion)
            return factor_cross_product(factors, result)
    

    def expand_requirement(requirement):
        factors = requirement['factors']
        expansion = factor_cross_product(factors.copy(), [{}])
        return expansion

    # (13) 
    satisfy_every_requirement = \
    And([
        And([
            Or([
                And([sample_factors[x][factor][level]
                    for factor, level in xr.items()]) 
                for x in samples]) 
            for xr in expand_requirement(r)])
        for r in requirements])
    l.debug("satisfy_every_requirement: %s", satisfy_every_requirement)
    constraints.append(satisfy_every_requirement)

#print(sample_factors_constraint)


    ## Factor level assignments are mutex
    factor_mutex = \
      And([
          And(
          And([
            Or(Not(sample_factors[x][factor_id][level1]),
                Not(sample_factors[x][factor_id][level2]))
            for level1 in factor['domain']
            for level2 in factor['domain'] if level2 != level1]),
          And([
             Or(Not(sample_factors[x][factor_id][level1]),
                Not(sample_factors_perp[x][factor_id]))
            for level1 in factor['domain']]))
          for factor_id, factor in factors.items()
          for x in samples])
    l.debug("factor_mutex: %s", factor_mutex)
    constraints.append(factor_mutex)

    ## tau mutex
    tau_mutex = \
      And([
          And(
          And([
            Or(Not(tau_symbols[x][a1]),
                Not(tau_symbols[x][a2]))
            for a1 in aliquots
            for a2 in aliquots if a1 != a2]),
          And([
             Or(Not(tau_symbols[x][a1]),
                Not(tau_symbols_perp[x]))
            for a1 in aliquots]))
          for x in samples])
    l.debug("tau_mutex: %s", tau_mutex)
    constraints.append(tau_mutex)

    ## Each tau(x) has a value
    tau_values = And([Or(Or([tau_symbols[x][a]
            for a in aliquots]),
            tau_symbols_perp[x])
        for x in samples])
    constraints.append(tau_values)

    ## Each sample factor has a value
    sample_factor_values = \
      And([Or(Or([sample_factors[x][factor_id][level1]
            for level1 in factor['domain']]),
            sample_factors_perp[x][factor_id])
        for factor_id, factor in factors.items()
        for x in samples])
    constraints.append(sample_factor_values)

    f = And(constraints)
    #l.debug("Constraints: %s", f)

    return variables, f

def generate_constraints1(inputs):
    """
    Generate constraints for plate layout encoding.
    """
    variables, values = generate_variables1(inputs)

    constraints = []

    #bounds = generate_bounds(inputs, variables, values)
    #constraints.append(bounds)
    
    samples = inputs['samples']
    factors = inputs['factors']
    containers = inputs['containers']
    requirements = inputs['requirements']

    aliquots = [a for c in containers for a in containers[c]['aliquots']]


#    tau_symbols = variables['tau_symbols']
    aliquot_factors = variables['aliquot_factors']
    sample_factors = variables['sample_factors']
    exp_factor = variables['exp_factor']
    batch_factor = variables['batch_factor']
    column_factor = variables['column_factor']

    ## CS
    def aliquot_can_satisfy_requirement(aliquot, requirement):
        """
        Are the aliquot properties consistent with requirement?
        Both are a conjunction of factor assignments, so make
        sure that aliquot is a subset of the requirement.
        """

        requirement_factors = [f['factor'] for f in requirement]
        requirement_levels = {f['factor']:f['values'] for f in requirement}
        for _, c in containers.items(): ## FIXME assumes that aliquots have unique names across containers
            #l.debug("Can sat? %s %s", aliquot, requirement)
            aliquot_properties = c['aliquots'][aliquot]
            for factor, level in aliquot_properties.items():
                #l.debug("checking: %s %s", factor, level)
                if factor in requirement_factors:
                    #l.debug("Is %s in %s", level, requirement_levels[factor])
                    if level not in requirement_levels[factor]:
                        #l.debug("no")
                        return False
            #l.debug("yes")
            return True
        #l.debug("no")
        return False ## Couldn't find a container with the aliquot satisfying requirement

    def mutex(levels):
          return And([Or(Not(level1_symbol), Not(level2_symbol))
                            for level1, level1_symbol in levels.items()
                            for level2, level2_symbol in levels.items() if level2 != level1])
    
    def cs_factor_level(factor, levels):
          return And(Or([level_symbol for level, level_symbol in levels.items()]), mutex(levels))
    
    def cs_factors_level(factors):
        return And([cs_factor_level(factor_id, levels)
                    for factor_id, levels in factors.items()])
    
    def cs_experiment_factors(exp_factor):
        return cs_factors_level(exp_factor)

    def cs_sample_factors(sample_factor, container_id, container, aliquot):
        return And([cs_factors_level(sample_factor[container_id][aliquot][sample])
                    for sample in samples[container_id][aliquot]])
            
    def cs_aliquot_factors(aliquot_factors, container_id, container, column):
        return And([And(cs_factors_level(aliquot_factors[container_id][aliquot]),
                        cs_sample_factors(sample_factors, container_id, container, aliquot))
                    for aliquot in column])
    
    def cs_column_factors(column_factor, container_id, container):
        def get_column_factors(column_factors, col):
            return { factor_id : { level : columns[col] for level, columns in levels.items() } for factor_id, levels in column_factors.items() }
        
        return And([And(cs_factors_level(get_column_factors(column_factor, column)),
                        cs_aliquot_factors(aliquot_factors, container_id, container, container['columns'][column]))
                    for column in container['columns']])

    
    def cs_batch_factors(batch_factors, containers):
        def get_batch_factors(batch_factors, container):
            return { factor_id : { level : containers[container] for level, containers in levels.items() } for factor_id, levels in batch_factors.items() }
        
        return And([And(cs_factors_level(get_batch_factors(batch_factors, container_id)),
                        cs_column_factors(column_factor, container_id, container))
                    for container_id, container in containers.items()])
          

    
    condition_space_constraint = \
      And(cs_experiment_factors(exp_factor),
         cs_batch_factors(batch_factor, containers)
          #cs_aliquot_factors(),
          #cs_sample_factors()
              )
    l.debug("CS: %s", condition_space_constraint)
    constraints.append(condition_space_constraint)
    

    # ALQ
    aliquot_properties_constraint = \
      And([
                   And([aliquot_factors[container_id][aliquot][factor][level] 
                        for factor, level in aliquot_properties.items()])
               for container_id, c in containers.items()
               for aliquot, aliquot_properties in c['aliquots'].items()
            ])
    #l.debug("aliquot_properties_constraint: %s", aliquot_properties_constraint)
    #constraints.append(aliquot_properties_constraint)



    def factor_cross_product(factors, cross_product):
        if len(factors) == 0:
            return cross_product
        else:
            result = []
            factor = factors.pop(0)
            for elt in cross_product:
                for value in factor['values']:
                    expansion = elt.copy()
                    expansion.update({factor['factor'] : value})
                    result.append(expansion)
            return factor_cross_product(factors, result)
    

    def expand_requirement(factors):
        #factors = requirement['factors']
        expansion = factor_cross_product(factors.copy(), [{}])
        return expansion


    def r_exp_factors(r):
        return [ f for f in r['factors'] if factors[f['factor']]['ftype'] == "experiment" ]
    def r_batch_factors(r):
        return [ f for f in r['factors'] if factors[f['factor']]['ftype'] == "batch" ]
    def r_column_factors(r):
        return [ f for f in r['factors'] if factors[f['factor']]['ftype'] == "column" ]
    def r_aliquot_factors(r):
        return [ f for f in r['factors'] if factors[f['factor']]['ftype'] == "aliquot" ]
    def r_sample_factors(r):
        return [ f for f in r['factors'] if factors[f['factor']]['ftype'] == "sample" ]
    
    def req_experiment_factors(r_exp_factors):
        return And([And([exp_factors[factor['factor']][level]
                        for level in factor['values']])
                    for factor in r_exp_factors])

    def req_sample_factors(r_sample_factors, samples, aliquot, container):
        cases = expand_requirement(r_sample_factors)
        l.debug("factors: %s, aliquots: %s", r_sample_factors, samples)
        l.debug("|cases| = %s, |samples| = %s", len(cases), len(samples))
        assert(len(cases) <= len(samples))
        clause = And([Or([And([sample_factors[container][aliquot][sample][factor][level]
                             for factor, level in case.items()])
                        for sample in samples])
                for case in cases])
        return clause

#            return And([And([Or([sample_factors[aliquot][sample][factor["factor"]][level]
#                                for sample in samples])
#                            for level in factor['values']])
#                        for factor in r_sample_factors])

   
    def req_aliquot_factors(r, r_aliquot_factors, container, aliquots):
        """
        Disjunction over aliquots, conjunction over factors and levels
        """
        cases = expand_requirement(r_aliquot_factors)
        l.debug("factors: %s, aliquots: %s", r_aliquot_factors, aliquots)
        l.debug("|cases| = %s, |aliquots| = %s", len(cases), len(aliquots))
        assert(len(cases) <= len(aliquots))
        clause = And([Or([And(And([aliquot_factors[container][aliquot][factor][level]
                                   for factor, level in case.items()]),
                              req_sample_factors(r_sample_factors(r), samples[container][aliquot], aliquot, container))
                        for aliquot in aliquots \
                              if aliquot_can_satisfy_requirement(aliquot, [{"factor" : factor, "values" : [level]}
                                                                               for factor, level in case.items()])])
                for case in cases])
        return clause            

    
    def req_column_factors(r, r_column_factors, container, columns):
        if len(r_column_factors) > 0:
            cases = expand_requirement(r_column_factors)
            assert(len(cases) <= len(columns))
            clause = And([Or([And(And([column_factor[factor][level][column]
                                   for factor, level in case.items()]),
                              req_aliquot_factors(r, r_aliquot_factors(r), container, columns[column]))
                        for column in columns])
                for case in cases])
            return clause            
        else:
            return req_aliquot_factors(r, r_aliquot_factors(r), container, [a for column in columns for a in columns[column]])
            
        
    def req_batch_factors(r, r_batch_factors, containers):
        """
        Satisfying a batch requirement requires having enough containers to 
        satisfy each combination
        """
        cases = expand_requirement(r_batch_factors)
        assert(len(cases) <= len(containers))

        clause = And([Or([And(And([batch_factor[factor][level][container_id]
                                   for factor, level in case.items()]),
                              req_column_factors(r, r_column_factors(r), container_id, container['columns']))
                        for container_id, container in containers.items()])
                    for case in cases])
        return clause            


    satisfy_every_requirement = \
    And([
        And(
            req_experiment_factors(r_exp_factors(r)),
            req_batch_factors(r, r_batch_factors(r), containers))
        for r in requirements])
    l.debug("satisfy_every_requirement: %s", satisfy_every_requirement)
    constraints.append(satisfy_every_requirement)

    ## Factor level assignments are mutex
#    factor_mutex = \
#      And([
#          And([
#            Or(Not(sample_factors[a][x][factor_id][level1]),
#                Not(sample_factors[a][x][factor_id][level2]))
#            for level1 in factor['domain']
#            for level2 in factor['domain'] if level2 != level1])
#          for factor_id, factor in factors.items()
#          for a in samples
#          for x in samples[a]])
    #l.debug("factor_mutex: %s", factor_mutex)
    #constraints.append(factor_mutex)


    ## Each sample factor has a value
 #   sample_factor_values = \
 #     And([Or([sample_factors[a][x][factor_id][level1]
 #           for level1 in factor['domain']])
 #       for factor_id, factor in factors.items()
 #       for a in samples
 #       for x in samples[a]])
    #constraints.append(sample_factor_values)

    f = And(constraints)
    #l.debug("Constraints: %s", f)

    return variables, f

def solve(input):
    """
    Convert input to encoding and invoke solver.  Return model if exists.
    """

    if not input['samples']:
        input['samples'] = [ "x{}".format(x) for x in range(0, 84) ]
    
    variables, constraints = generate_constraints(input)
    model = get_model(constraints)
    return model, variables

def solve1(input):
    """
    Convert input to encoding and invoke solver.  Return model if exists.
    """

    if not input['samples']:
        containers = input['containers']
        input['samples'] = {
            c: {
                a : {
                    x : "x{}_{}_{}".format(x, a, c) for x in range(0, 10) }
                for a in containers[c]['aliquots'] }
            for c in containers }
    
    variables, constraints = generate_constraints1(input)
    model = get_model(constraints)
    return model, variables



def get_model_pd(model, variables):


    experiment_df = pd.DataFrame()
    batch_df = pd.DataFrame()
    column_df = pd.DataFrame()
    aliquot_df = pd.DataFrame()
    sample_df = pd.DataFrame()

    def info_to_df(info):
        info_df = pd.DataFrame()
        return info_df.append(info, ignore_index=True)

    def merge_info_df(df, info, on):
        if len(df) > 0:
            #l.debug("Merge L: %s", df)
            #l.debug("Merge R: %s", info_to_df(info))
            #onon = df.columns.intersection(info_to_df(info).columns)
            #l.debug("onon %s", onon.values)
            #df = df.merge(info_to_df(info),  how='outer', on=list(onon.values))#, suffixes=('', '_y'))
            if on:
                df = df.set_index(on).combine_first(info_to_df(info).set_index(on)).reset_index()
            else:
                df = df.combine_first(info_to_df(info))
            #to_drop = [x for x in df if x.endswith('_y')]
            #df = df.drop(to_drop, axis=1)

            #l.debug("Merge O: %s", df)
        else:
            df = df.append(info, ignore_index=True)
        return df
    
    for var, value in model:
        if value.is_true():
            if var in variables['reverse_index']:
                l.debug("{} = {}".format(var, value))
                info = variables['reverse_index'][var]
               # l.debug("info = %s", info)
                if info['type'] == 'aliquot':
                    aliquot_df = merge_info_df(aliquot_df, info, ["aliquot", "container"])
                elif info['type'] == 'sample':
                    sample_df = merge_info_df(sample_df, info, "sample")
                elif info['type'] == 'batch':
                    batch_df = merge_info_df(batch_df, info, "container")
                elif info['type'] == 'columm':
                    column_df = merge_info_df(column_df, info, "column")
                elif info['type'] == 'experiment':
                    l.debug("info: %s", info)
                    experiment_df = experiment_df(experiment_df, info, None)
            
    l.debug("aliquot_df %s", aliquot_df)
    l.debug("sample_df %s", sample_df)
    l.debug("column_df %s", column_df)
    l.debug("batch_df %s", batch_df)
    l.debug("experiment_df %s", experiment_df)
    df = aliquot_df.drop(columns=['type']).merge(sample_df.drop(columns=['type']), on=["aliquot", "container"])
    if len(column_df) > 0:
        df = df.merge(column_df.drop(columns=['type']), on=["container"])
    df = df.merge(batch_df.drop(columns=['type']), on=["container"])
    if len(experiment_df) > 0:
        df['key'] = 0
        experiment_df['key'] = 0
        df = df.merge(experiment_df, on=['key']).drop(columns=['key'])

    l.debug("df %s", df)               
    #l.debug(aliquot_df.loc[aliquot_df.aliquot=='a5'])
    #l.debug(sample_df.loc[sample_df.aliquot=='a5'])
    #df = experiment_df
    #aliquot_df['key'] = 0
    #l.debug(aliquot_df)
    #l.debug(df)
    #df = df.merge(aliquot_df, on=['container'])
            
#    df = aliquot_df
#    l.debug(df)
#    df = df.drop(columns=['type'])
#    batch_df = batch_df.drop(columns=['type'])
    
#    l.debug(batch_df)
#    df = df.merge(batch_df, on='container')
        
#    df = df.sort_values(by=['aliquot'])
    #l.debug(df.loc[df.aliquot=='a5'])
    df.to_csv("dan.csv")

    
