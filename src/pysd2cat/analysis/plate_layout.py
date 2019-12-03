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
    variables['tau_symbols'] = \
      {
          a : {
              x : Symbol("tau_{}".format(a[x]))
              for x in samples[a] 
            }
            for a in samples

      }
    for aliquot in variables['tau_symbols']:
        for sample in variables['tau_symbols'][aliquot]:
            var = variables['tau_symbols'][aliquot][sample]
            if var not in variables['reverse_index']:                
                variables['reverse_index'][var] = {}
            variables['reverse_index'][var].update({"sample": "x{}_{}".format(sample, aliquot), "aliquot" : aliquot})
            
    variables['sample_factors'] = \
      {
          a : {
              sample: {
                factor_id : {
                    level : Symbol("{}(x{}_{})={}".format(factor_id, sample, a, level))
                    for level in factor['domain']
                    }
                  for factor_id, factor in factors.items()
                }
                for sample in samples[a]
            }
        for a in samples
      }
    for aliquot in variables['sample_factors']: 
        for sample in variables['sample_factors'][aliquot]:
            for factor_id in variables['sample_factors'][aliquot][sample]:
                for level in variables['sample_factors'][aliquot][sample][factor_id]:
                    var = variables['sample_factors'][aliquot][sample][factor_id][level]
                    if var not in variables['reverse_index']:                
                        variables['reverse_index'][var] = {}
                    variables['reverse_index'][var].update({"aliquot" : aliquot, "sample" :  "x{}_{}".format(sample, aliquot), factor_id : level})

        
    values = {}

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


    tau_symbols = variables['tau_symbols']
    sample_factors = variables['sample_factors']
    exp_factor = variables['exp_factor']
    batch_factor = variables['batch_factor']
    column_factor = variables['column_factor']
    
    
    
    # (4)
    requirements_constraint = \
      And([Implies(tau_symbols[aliquot][x],
                   Or([
                       And([
                           Or([sample_factors[aliquot][x][f['factor']][level]                           
                               for level in f['values']])
                           for f in r["factors"]]) 
                       for r in requirements]))
            for aliquot in samples
            for x in samples[aliquot]])
    l.debug("requirements_constraint: %s", requirements_constraint)
    constraints.append(requirements_constraint)

    # (5)
    aliquot_properties_constraint = \
      And([Implies(tau_symbols[aliquot][x],
                   And([sample_factors[aliquot][x][factor][level] 
                        for factor, level in aliquot_properties.items()]))
               for _, c in containers.items()
               for aliquot, aliquot_properties in c['aliquots'].items()
               for x in samples[aliquot]

            ])
    l.debug("aliquot_properties_constraint: %s", aliquot_properties_constraint)
    constraints.append(aliquot_properties_constraint)

    # (6)
    experiment_factors_constraint = \
      And([Implies(tau_symbols[aliquot][x],
                   And([
                       Or([And(sample_factors[x][factor_id][level],
                               exp_factor[factor_id][level])
                           for level in factor['domain']
                            ])
                       for factor_id, factor in factors.items() if factor["ftype"] == "experiment"
                     ]))
         for aliquot in samples
         for x in samples[aliquot]])
    l.debug("experiment_factors_constraint: %s", experiment_factors_constraint)
    constraints.append(experiment_factors_constraint)

    # (7)
    batch_factors_constraint = \
      And([ 
        Implies(tau_symbols[aliquot][x],
                And([
                     Or([And(sample_factors[aliquot][x][factor_id][level],
                             batch_factor[factor_id][level][container_id])
                        for level in factor['domain']
                        ])
                     for factor_id, factor in factors.items() if factor["ftype"] == "batch"
                     ]))
        for container_id, container in containers.items()
        for aliquot, aliquot_properties in container['aliquots'].items()
        for x in samples[aliquot]
        ])
    l.debug("batch_factors_constraint: %s", batch_factors_constraint)
    constraints.append(batch_factors_constraint)

    # (8)
    sample_factors_constraint = \
    And([
        Implies(And(tau_symbols[a][x], tau_symbols[a][xp]),                    
                And([
                    Or([And(sample_factors[a][x][factor_id][level],
                            sample_factors[a][xp][factor_id][level])
                        for level in factor['domain']
                        ])
                     for factor_id, factor in factors.items()  if factor["ftype"] == "sample"]))
        for a in samples
        for xp in samples[a]
        for x in samples[a]])
    l.debug("sample_factors_constraint: %s", sample_factors_constraint)
    constraints.append(sample_factors_constraint)

    # (9)
    if len(column_factor) > 0:
        column_factors_constraint = \
        And([ 
            Implies(tau_symbols[a][x], 
                    And([
                        Or([And(sample_factors[a][x][factor_id][level],
                                column_factor[factor_id][level][column_id])
                            for level in factor['domain']
                            ])
                        for factor_id, factor in factors.items() if factor["ftype"] == "column"]))    
            for container_id, container in containers.items()
            for column_id, column in container['columns'].items()
            for a in column
            for x in samples[a]
                        
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

    def aliquot_can_satisfy_requirement(aliquot, requirement):
        """
        Are the aliquot properties consistent with requirement?
        Both are a conjunction of factor assignments, so make
        sure that aliquot is a subset of the requirement.
        """
        for _, c in containers.items(): ## FIXME assumes that aliquots have unique names across containers
            aliquot_properties = c['aliquots'][aliquot]
            for factor, level in aliquot_properties.items():
                if factor in requirement:
                    if requirement[factor] != level:
                        return False
            return True
        return False ## Couldn't find a container with the aliquot satisfying requirement
    
    # (13) 
    satisfy_every_requirement = \
    And([
        And([
            Or([
                And([sample_factors[a][x][factor][level]
                    for factor, level in xr.items()])
                for a in samples if aliquot_can_satisfy_requirement(a, xr)
                for x in samples[a]]) 
            for xr in expand_requirement(r)])
        for r in requirements])
    l.debug("satisfy_every_requirement: %s", satisfy_every_requirement)
    constraints.append(satisfy_every_requirement)

    ## Factor level assignments are mutex
    factor_mutex = \
      And([
          And([
            Or(Not(sample_factors[a][x][factor_id][level1]),
                Not(sample_factors[a][x][factor_id][level2]))
            for level1 in factor['domain']
            for level2 in factor['domain'] if level2 != level1])
          for factor_id, factor in factors.items()
          for a in samples
          for x in samples[a]])
    l.debug("factor_mutex: %s", factor_mutex)
    constraints.append(factor_mutex)


    ## Each sample factor has a value
    sample_factor_values = \
      And([Or([sample_factors[a][x][factor_id][level1]
            for level1 in factor['domain']])
        for factor_id, factor in factors.items()
        for a in samples
        for x in samples[a]])
    constraints.append(sample_factor_values)

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
        aliquots = [a for c in containers for a in containers[c]['aliquots']]
        input['samples'] = { a : { x : "x{}_{}".format(x, a) for x in range(0, 1) } for a in aliquots }
    
    variables, constraints = generate_constraints1(input)
    model = get_model(constraints)
    return model, variables



def get_model_pd(model, variables):

    sample_info = {}
    for var, value in model:
        if value.is_true():
            if var in variables['reverse_index']:
                info = variables['reverse_index'][var]
                sample = info['sample']
                if sample not in sample_info:
                    sample_info[sample] = {}
                sample_info[sample].update(info)
            l.debug("{} = {}".format(var, value))
    l.debug(sample_info)
    df = pd.DataFrame()
    for sample, info in sample_info.items():
        df = df.append(info, ignore_index=True)
    l.debug(df.sort_values(by=['aliquot']))

    
