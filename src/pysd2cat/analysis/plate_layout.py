from pysmt.shortcuts import Symbol, And, Or, Not, Implies, Equals, Iff, is_sat, get_model, GT, GE, LT, LE, Int, Real, String, TRUE, ExactlyOne, get_unsat_core
from pysmt.typing import INT, StringType, REAL
from functools import reduce

from pysd2cat.analysis.plate_layout_utils import get_samples_from_condition_set, get_column_name, get_column_factors, get_column_id, get_row_name, get_row_factors, get_row_id, get_aliquot_row, get_aliquot_col

from pysd2cat.analysis.container_data_conversion import get_strain_count

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

    def get_factor_symbols(factor, prefix, var=None, constraints=None):
        if factor['dtype'] == "str":
            if var is not None and constraints and var in constraints and constraints[var]:
                #l.debug("setting levels of %s %s %s", var, factor['name'],  constraints[var][factor['name']])
                levels = [ constraints[var][factor['name']] ]
            else:
                levels = factor['domain']
            return  {
                level : Symbol("{}={}".format(prefix, level))
                for level in levels
            }
        else:
            # Cannot filter values here because its an real that is yet to be assigned
            return Symbol(prefix, REAL)

    variables['aliquot_factors'] = \
      {
          c: {
              a : {
                  factor_id : get_factor_symbols(factor, "{}({}_{})".format(factor_id, a, c))
                  for factor_id, factor in factors.items() if factor['ftype'] == "aliquot"                
              }
              for a in samples[c]
          }
          for c in containers
      }
    #l.info( variables['aliquot_factors'])


    
    variables['sample_factors'] = \
      {
          c : {
              a : {
                  sample: {
                      factor_id : get_factor_symbols(factor,
                                                     "{}(x{}_{}_{})".format(factor_id, sample, a, c),
                                                     var=sample,
                                                     constraints=inputs['sample_types'])
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
                    if factors[factor_id]['dtype'] == "str":
                        for level in variables['sample_factors'][container][aliquot][sample][factor_id]:
                            var = str(variables['sample_factors'][container][aliquot][sample][factor_id][level])
                            if var not in variables['reverse_index']:                
                                variables['reverse_index'][var] = {}
                            variables['reverse_index'][var].update({"type" : "sample", "aliquot" : aliquot, "sample" :  "x{}_{}_{}".format(sample, aliquot, container), "container" : container, factor_id : level})
                    else:
                        var = str(variables['sample_factors'][container][aliquot][sample][factor_id])
                        if var not in variables['reverse_index']:                
                            variables['reverse_index'][var] = {}
                        variables['reverse_index'][var].update({"type" : "sample", "aliquot" : aliquot, "sample" :  "x{}_{}_{}".format(sample, aliquot, container), "container" : container, factor_id : None})

            for factor_id in variables['aliquot_factors'][container][aliquot]:
                if factors[factor_id]['dtype'] == "str":
                    for level in variables['aliquot_factors'][container][aliquot][factor_id]:
                        var = str(variables['aliquot_factors'][container][aliquot][factor_id][level])
                        if var not in variables['reverse_index']:                
                            variables['reverse_index'][var] = {}
                        column = [col for col, aliquots in containers[container]['columns'].items() if aliquot in aliquots][0]
                        row = [row for row, aliquots in containers[container]['rows'].items() if aliquot in aliquots][0]
                        variables['reverse_index'][var].update({"type" : "aliquot", "aliquot" : aliquot, "container" : container, factor_id : level, "column" : get_column_name(column, container), "column_id" : get_column_id(column), "row" : get_row_name(row, container)})
                else:
                    var = str(variables['aliquot_factors'][container][aliquot][factor_id])
                    if var not in variables['reverse_index']:                
                        variables['reverse_index'][var] = {}
                    column = [col for col, aliquots in containers[container]['columns'].items() if aliquot in aliquots][0]
                    row = [row for row, aliquots in containers[container]['rows'].items() if aliquot in aliquots][0]
                    variables['reverse_index'][var].update({"type" : "aliquot", "aliquot" : aliquot, "container" : container, factor_id : level, "column" : get_column_name(column, container), "column_id" : get_column_id(column), "row" : get_row_name(row, container)})

        
    values = {}

    variables['exp_factor'] = \
      {
          factor_id : get_factor_symbols(factor, "{}()".format(factor_id))
          for factor_id, factor in factors.items() if factor['ftype'] == "experiment"
      }
    for exp_factor in variables['exp_factor']:
        if factors[exp_factor]['dtype'] == "str":
            for level in variables['exp_factor'][exp_factor]:
                var = str(variables['exp_factor'][exp_factor][level])
                if var not in variables['reverse_index']:
                    variables['reverse_index'][var] = {}
                variables['reverse_index'][var].update({"type" : "experiment", exp_factor : level})
        else:
            var = str(variables['exp_factor'][exp_factor])
            if var not in variables['reverse_index']:
                variables['reverse_index'][var] = {}
            variables['reverse_index'][var].update({"type" : "experiment", exp_factor : None})


    container_assignment = inputs['container_assignment']
    l.info("container_assignment: %s", container_assignment)
            
    variables['batch_factor'] = \
      {
          container : {
              factor_id : get_factor_symbols(factor, "{}({})".format(factor_id, container), container, container_assignment)
              #factor_id : get_factor_symbols(factor, "{}({})".format(factor_id, container))
              for factor_id, factor in factors.items() if factor['ftype'] == "batch"
              }
              for container in containers                          
      }
    l.debug("batch_factors %s", variables['batch_factor'])
      
    for container  in variables['batch_factor']:
        for batch_factor  in variables['batch_factor'][container]:
            if factors[batch_factor]['dtype'] == "str":
                for level in variables['batch_factor'][container][batch_factor]:
                    var = str(variables['batch_factor'][container][batch_factor][level])
                    if var not in variables['reverse_index']:
                        variables['reverse_index'][var] = {}
                    variables['reverse_index'][var].update({"type" : "batch", "container" : container, batch_factor : level})
            else:
                var = str(variables['batch_factor'][container][batch_factor])
                if var not in variables['reverse_index']:
                    variables['reverse_index'][var] = {}
                variables['reverse_index'][var].update({"type" : "batch", "container" : container, batch_factor : None})

    ## Variables used when choose to ignore the column factor
    variables['na_column_factors'] = \
      {
          get_column_name(col, container_id) : {
            factor_id : Symbol("{}_is_na({}_{})".format(factor_id, col, container_id))
            for factor_id, factor in factors.items() if factor['ftype'] == "column"
            }
        for container_id, container in containers.items()
        for col in container['columns']    
      }
    for column in variables['na_column_factors']:
        #l.debug("Reversing: %s", column)
        for column_factor in variables['na_column_factors'][column]:
            var = str(variables['na_column_factors'][column][column_factor])
            if var not in variables['reverse_index']:
                variables['reverse_index'][var] = {}
            container = [ c for c in containers if c in column][0] ## assumes container_id is in column name
            variables['reverse_index'][var].update({"type" : "na_column", "column" : column, "container" : container, column_factor : "NA", "applicable" : False})
                    
    variables['column_factor'] = \
      {
          get_column_name(col, container_id) : {
            factor_id : get_factor_symbols(factor, "{}({}_{})".format(factor_id, col, container_id))
            for factor_id, factor in factors.items() if factor['ftype'] == "column"
            }
        for container_id, container in containers.items()
        for col in container['columns']    
      }
    #l.info("column_factors variables: %s", variables['column_factor'])
    for column in variables['column_factor']:
        #l.debug("Reversing: %s", column)
        for column_factor in variables['column_factor'][column]:
            #l.debug("Reversing factor: %s %s", column_factor,  factors[column_factor]['dtype'])
            if factors[column_factor]['dtype'] == "str":
                for level in variables['column_factor'][column][column_factor]:
                    var = str(variables['column_factor'][column][column_factor][level])
                    if var not in variables['reverse_index']:
                        variables['reverse_index'][var] = {}
                    #container = [ c for c in containers if column in containers[c]['columns']][0]
                    container = [ c for c in containers if c in column][0] ## assumes container_id is in column name
                    variables['reverse_index'][var].update({"type" : "column", "column" : column, "container" : container, column_factor : level})
            else:
                var = str(variables['column_factor'][column][column_factor])
                if var not in variables['reverse_index']:
                    variables['reverse_index'][var] = {}
                #container = [ c for c in containers if column in containers[c]['columns']][0]
                container = [ c for c in containers if c in column][0] ## assumes container_id is in column name
                variables['reverse_index'][var].update({"type" : "column", "column" : column, "container" : container, column_factor : None})
                #l.debug("Reverse %s %s", var, variables['reverse_index'][var])

    variables['row_factor'] = \
      {
          get_row_name(row, container_id) : {
            factor_id : get_factor_symbols(factor, "{}({}_{})".format(factor_id, row, container_id))
            for factor_id, factor in factors.items() if factor['ftype'] == "row"
            }
        for container_id, container in containers.items()
        for row in container['rows']    
      }
    #l.debug("row_factors variables: %s", variables['row_factor'])
    for row in variables['row_factor']:
        #l.debug("Reversing: %s", row)
        for row_factor in variables['row_factor'][row]:
            #l.debug("Reversing factor: %s %s", row_factor,  factors[row_factor]['dtype'])
            if factors[row_factor]['dtype'] == "str":
                for level in variables['row_factor'][row][row_factor]:
                    var = str(variables['row_factor'][row][row_factor][level])
                    if var not in variables['reverse_index']:
                        variables['reverse_index'][var] = {}
                    #container = [ c for c in containers if row in containers[c]['rows']][0]
                    container = [ c for c in containers if c in row][0] ## assumes container_id is in row name
                    variables['reverse_index'][var].update({"type" : "row", "row" : row, "container" : container, row_factor : level})
            else:
                var = str(variables['row_factor'][row][row_factor])
                if var not in variables['reverse_index']:
                    variables['reverse_index'][var] = {}
                #container = [ c for c in containers if row in containers[c]['rows']][0]
                container = [ c for c in containers if c in row][0] ## assumes container_id is in row name
                variables['reverse_index'][var].update({"type" : "row", "row" : row, "container" : container, row_factor : None})
                #l.debug("Reverse %s %s", var, variables['reverse_index'][var])

    #l.debug("Variables: %s", variables)
    #l.debug("Values: %s", values)
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
    row_factor = variables['row_factor']

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
    #l.debug("satisfy_every_requirement: %s", satisfy_every_requirement)
    constraints.append(satisfy_every_requirement)



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
    na_column_factors = variables['na_column_factors']
    row_factor = variables['row_factor']

    container_assignment = inputs['container_assignment']
    
    ## CS
    def aliquot_can_satisfy_requirement(aliquot, container_id, requirement):
        """
        Are the aliquot properties consistent with requirement?
        Both are a conjunction of factor assignments, so make
        sure that aliquot is a subset of the requirement.
        """

        requirement_factors = [f['factor'] for f in requirement]
        requirement_levels = {f['factor']:f['values'] for f in requirement}
        c = containers[container_id]
        #l.debug("Can sat? %s %s %s", container_id, aliquot, requirement)
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
        #return False ## Couldn't find a container with the aliquot satisfying requirement

    def mutex(levels):
          return 
#          return And([Or(Not(level1_symbol), Not(level2_symbol))
#                            for level1, level1_symbol in levels.items()
#                            for level2, level2_symbol in levels.items() if level2 != level1])
    
    def cs_factor_level(levels):
          return  ExactlyOne(levels)

    def cs_factor_bounds(factor_id, symbol, levels):
        return And(GE(symbol, Real(levels[0])),
                   LE(symbol, Real(levels[1])))
      
    def cs_factors_level(factor_symbols, var=None, constraints=None, na_factors=None):
        factor_clauses = []
        for factor_id, symbols in factor_symbols.items():

            ## Get symbol for case where factor is not applicable.
            na_level = None
            if na_factors is not None and factor_id in na_factors:
                na_level = na_factors[factor_id]
            
            if factors[factor_id]['dtype'] == "str":
                # Already filtered the levels for Bools
                if na_level:
                    levels = [y for x, y in symbols.items()] + [na_level]
                else:
                    levels = [y for x, y in symbols.items()]
                factor_clauses.append(cs_factor_level(levels))
            else:
                # Need to filter levels for ints
                if var is not None and constraints and var in constraints and constraints[var]:
                    #l.debug("setting levels of %s %s %s", var, factor_id,  constraints[var][factor_id])
                    levels = [constraints[var][factor_id], constraints[var][factor_id]]
                else:
                    levels = factors[factor_id]['domain']
                if na_level:
                    factor_clauses.append(Or(na_level, cs_factor_bounds(factor_id, symbols, levels)))
                else:
                    factor_clauses.append(cs_factor_bounds(factor_id, symbols, levels))
                    
        return And(factor_clauses)
    
    def cs_experiment_factors(exp_factor):
        return cs_factors_level(exp_factor)

    def cs_sample_factors(sample_factor, container_id, container, aliquot):
        return And([cs_factors_level(sample_factor[container_id][aliquot][sample],
                                     var=sample,
                                     constraints=inputs['sample_types'])
                    for sample in samples[container_id][aliquot]])
            
    def cs_aliquot_factors(aliquot_factors, container_id, container, column):
        return And([And(cs_factors_level(aliquot_factors[container_id][aliquot]),
                        cs_sample_factors(sample_factors, container_id, container, aliquot))
                    for aliquot in column])
    
    def cs_column_factors(column_factor, container_id, container):
        
        clause = And(
                     ## Each Column has factors with values in their domain or are NA
                     And([cs_factors_level(get_column_factors(column_factor, column, container_id)
                                               ,
                                           na_factors=na_column_factors[get_column_name(column, container_id)]
                                               )
                       for column in container['columns']])
                       ,
                     ## For each factor, either all columns in container are NA or not (e.g., cannot induce at different times)
                     And([Or(And([na_column_factors[get_column_name(column, container_id)][factor_id]
                                 for column in container['columns']]),
                             And([Not(na_column_factors[get_column_name(column, container_id)][factor_id])
                              for column in container['columns']]))
                         for factor_id, factor in factors.items() if factor['ftype'] == "column"])
                    )
        #l.debug("column clause: %s", clause)
        return clause

    def cs_row_factors(row_factor, container_id, container):
        
        clause = And([And(cs_factors_level(get_row_factors(row_factor, row, container_id))
                        )
                    for row in container['rows']])
        #l.debug("column clause: %s", clause)
        return clause

    
    def cs_batch_factors(batch_factors, containers):
        def get_batch_factors(batch_factors, container):
            return batch_factors[container]
            #return { factor_id : { level : containers[container] for level, containers in levels.items() } for factor_id, levels in batch_factors.items() }
        
        return And([And(cs_factors_level(get_batch_factors(batch_factors, container_id), container_id, container_assignment),
                        cs_column_factors(column_factor, container_id, container),
                        cs_row_factors(row_factor, container_id, container),
                        cs_aliquot_factors(aliquot_factors, container_id, container, container['aliquots'])
                        )
                    for container_id, container in containers.items()])
          

    
    condition_space_constraint = \
      And(
          cs_experiment_factors(exp_factor),
          cs_batch_factors(batch_factor, containers)
         )
    #l.debug("CS: %s", condition_space_constraint)
    constraints.append(condition_space_constraint)
    
    
    #l.info(containers)
    # ALQ

    def get_req_const(factors_of_type, factor_id, level):            
        if factors[factor_id]['dtype'] == "str":
            pred = factors_of_type[factor_id][level]
        else:
            pred = Equals(factors_of_type[factor_id], Real(level))
        return pred
    
    aliquot_properties_constraint = \
      And([
          And(
            ## Every aliquot must satisfy the aliquot factors defined by the container
            And([get_req_const(aliquot_factors[container_id][aliquot], factor, level)
                 for factor, level in aliquot_properties.items() if factor in aliquot_factors[container_id][aliquot]]),
                 #,
            ## Every column factor implied by the container aliquots is satisfied
            And([get_req_const(column_factor[get_column_name(get_aliquot_col(aliquot, c), container_id)], factor, level)
                 for factor, level in aliquot_properties.items() if factor in column_factor[get_column_name(get_aliquot_col(aliquot, c), container_id)]]),
            ## Every row factor implied by the container aliquots is satisfied
            And([get_req_const(row_factor[get_row_name(get_aliquot_row(aliquot, c), container_id)], factor, level)
                 for factor, level in aliquot_properties.items() if factor in row_factor[get_row_name(get_aliquot_row(aliquot, c), container_id)]])
           )
           for container_id, c in containers.items()
           for aliquot, aliquot_properties in c['aliquots'].items()
        ])
    
    #l.debug("aliquot_properties_constraint: %s", aliquot_properties_constraint)
    constraints.append(aliquot_properties_constraint)

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
    def r_row_factors(r):
        return [ f for f in r['factors'] if factors[f['factor']]['ftype'] == "row" ]
    def r_aliquot_factors(r):
        return [ f for f in r['factors'] if factors[f['factor']]['ftype'] == "aliquot" ]
    def r_sample_factors(r):
        return [ f for f in r['factors'] if factors[f['factor']]['ftype'] == "sample" ]


    def get_req_const(factors_of_type, factor_id, level):            
        if factors[factor_id]['dtype'] == "str":
            pred = factors_of_type[factor_id][level]
        else:
            pred = Equals(factors_of_type[factor_id], Real(level))
        return pred
    
    def req_experiment_factors(r_exp_factors):
        return And([And([get_req_const(exp_factors, factor['factor'], level)
                        for level in factor['values']])
                    for factor in r_exp_factors])

    def req_sample_factors(r_sample_factors, samples, aliquot, container):
        cases = expand_requirement(r_sample_factors)
        #l.debug("factors: %s, aliquots: %s", r_sample_factors, samples)
        #l.debug("|cases| = %s, |samples| = %s", len(cases), len(samples))
        assert(len(cases) <= len(samples))

        def sample_consistent_with_case(sample, case):
            #l.debug("sample: %s case: %s sample_types %s", sample, case, inputs['sample_types'])
            if sample in inputs['sample_types'] and inputs['sample_types'][sample]:
                for factor_id, level in case.items():
                    if factor_id in inputs['sample_types'][sample] and inputs['sample_types'][sample][factor_id] != level:
                        return False
            return True
        
        clause = And([Or([And([get_req_const(sample_factors[container][aliquot][sample], factor_id, level)
                             for factor_id, level in case.items()])
                        for sample in samples if  sample_consistent_with_case(sample, case)])
                for case in cases])
        return clause
   
    def req_aliquot_factors(r, r_aliquot_factors, containers):
        """
        Disjunction over aliquots, conjunction over factors and levels
        """
        req_batch_factors = r_batch_factors(r)
        reqs = r_aliquot_factors +  r_column_factors(r) + r_row_factors(r) + req_batch_factors
        cases = expand_requirement(reqs)
        r_aliquot_factors_ids = [x['factor'] for x in r_aliquot_factors]
        r_batch_factor_ids = [x['factor'] for x in req_batch_factors]
        #aliquots = container['aliquots']
        #l.debug("factors: %s, aliquots: %s", r_aliquot_factors, aliquots)
        #l.debug("|cases| = %s, |aliquots| = %s", cases, len(aliquots))
        #assert(len(cases) <= len(aliquots))
        
        ## Is it an unsatisfiable requirement?
        for case in cases:
            possible_aliquots = []
            for container_id, container in containers.items():
                for aliquot in container['aliquots']:
                    if aliquot_can_satisfy_requirement(aliquot, container_id, [{"factor" : factor, "values" : [level]}
                                                                               for factor, level in case.items()]):
                        possible_aliquots.append(aliquot)
            if len(possible_aliquots) == 0:
                l.warning("Requirement cannot be satisfied with containers: %s", case)
            assert(len(possible_aliquots) > 0)

        
        clause = \
          And([ ## Must satisfy all cases
              Or([ ## A container must satisfy the case
                  Or([ ## An aliquot in the container must satisfy case
                      ## Aliquot must satisfy:
                      And(
                          ## Satisfy aliquot factors
                          And([get_req_const(aliquot_factors[container_id][aliquot], factor_id, level)
                                   for factor_id, level in case.items() if factor_id in r_aliquot_factors_ids]),
                          ## Batch factors for container of aliquot
                          And([get_req_const(batch_factor[container_id], factor_id, level)            
                                   for factor_id, level in case.items() if factor_id in r_batch_factor_ids]),
                          ## Column factors for column of aliquot
                          req_column_factors(r, r_column_factors(r), [case], container_id, [get_aliquot_col(aliquot, container)]),
                          ## Row factors for row of aliquot
                          req_row_factors(r, r_row_factors(r), [case], container_id, [get_aliquot_row(aliquot, container)]),
                          ## Sample factors for aliquot
                          req_sample_factors(r_sample_factors(r), samples[container_id][aliquot], aliquot, container_id)
                         )
                        for aliquot in container['aliquots'] \
                              if aliquot_can_satisfy_requirement(aliquot, container_id, [{"factor" : factor, "values" : [level]}
                                                                               for factor, level in case.items()])])
                        for container_id, container in containers.items()])
                for case in cases])
        return clause            


    
    def req_column_factors(r, r_column_factors, cases, container_id, columns):
        """
        At least one of the columns must satisfy the factors in each case.
        """
        if len(r_column_factors) > 0:
            #cases = expand_requirement(r_column_factors)
            #assert(len(cases) <= len(columns))
            r_column_factors_ids = [x['factor'] for x in r_column_factors]
            clause = And([Or([
                              And([And(get_req_const(column_factor[get_column_name(column, container_id)], factor_id, level),
                                       Not(na_column_factors[get_column_name(column, container_id)][factor_id]))
                                   for factor_id, level in case.items() if factor_id in r_column_factors_ids and level != "NA" ]
                                   +
                                   [na_column_factors[get_column_name(column, container_id)][factor_id]
                                    for factor_id, level in case.items() if factor_id in r_column_factors_ids and level == "NA"]
                                      )
                        for column in columns])
                for case in cases])
            return clause            
        else:
            return And()
            
    def req_row_factors(r, r_row_factors, cases, container_id, rows):
        if len(r_row_factors) > 0:
            #cases = expand_requirement(r_row_factors)
            #assert(len(cases) <= len(rows))
            r_row_factors_ids = [x['factor'] for x in r_row_factors]
            clause = And([Or([And(And([get_req_const(row_factor[get_row_name(row, container_id)], factor_id, level)
                                   for factor_id, level in case.items() if factor_id in r_row_factors_ids])
                              
                                      )
                        for row in rows])
                for case in cases])
            return clause            
        else:
            return And()


        
    def req_batch_factors(r, r_batch_factors, containers):
        """
        Satisfying a batch requirement requires having enough containers to 
        satisfy each combination
        """
        cases = expand_requirement(r_batch_factors)
        assert(len(cases) <= len(containers))

        def consitent_with_container(case, container_id, container_assignment):
            if container_assignment:
                l.debug("Is consistent_with_container? %s %s", case, container_assignment[container_id])
                for factor_id, level in case.items():
                    if factor_id in container_assignment[container_id] and container_assignment[container_id][factor_id] != level:
                        return False
            return True
        
        clause = \
          And([
              #And(
                  ## There exists a container that satisfies batch factors in case
                  #Or([
                  #    # Batch factors
                  #    And([get_req_const(batch_factor[container_id], factor_id, level)            
                  #         for factor_id, level in case.items()])                         
                  #    for container_id, container in containers.items() if consitent_with_container(case, container_id, container_assignment)]),
                  ## There exists aliquots that satisfy the requirement
                  ## FIXME if there are no batch factors in case, then make sure that aliquots across all containers satisfy requirement
                  ##       otherwise, ensure that the aliquots satisfy the case are from the container satisfying the case.
                  req_aliquot_factors(r, r_aliquot_factors(r), containers)
                  #)
                for case in cases])
        return clause            

    l.info("Encoding %s requirements", len(requirements))

    satisfy_every_requirement = \
    And([
        And(
            req_experiment_factors(r_exp_factors(r)),
            req_batch_factors(r, r_batch_factors(r), containers)
            )
        for r in requirements])
    #l.debug("satisfy_every_requirement: %s", satisfy_every_requirement)
    constraints.append(satisfy_every_requirement)


    ## Column reagents are set to zero if every aliquot in the column is empty
    if "None" in factors['strain']['domain']:
        no_reagents_for_empty_columns = \
          And([And([Implies(And([aliquot_factors[container_id][aliquot]['strain']["None"]
                               for aliquot in column_aliquots]),
                          And([Equals(col_factor_var , Real(0.0))
                               for col_factor_id, col_factor_var in column_factor[get_column_name(column_id, container_id)].items()]))
               for column_id, column_aliquots in container['columns'].items()])
              for container_id, container in containers.items()])
        constraints.append(no_reagents_for_empty_columns)

        ## Row reagents are set to zero if every aliquot in the row is empty
        no_reagents_for_empty_rows = \
          And([And([Implies(And([aliquot_factors[container_id][aliquot]['strain']["None"]
                               for aliquot in row_aliquots]),
                          And([Equals(row_factor_var , Real(0.0))
                               for row_factor_id, row_factor_var in row_factor[get_row_name(row_id, container_id)].items()]))
               for row_id, row_aliquots in container['rows'].items()])
              for container_id, container in containers.items()])
        constraints.append(no_reagents_for_empty_rows)
      
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

def get_sample_types(sample_factors, requirements):
    """
    Get the number of unique assignments to the sample factors in the requirements
    """
    experiment_design = pd.DataFrame()

    for condition_set in requirements:       
        samples = get_samples_from_condition_set(sample_factors, condition_set)
        #l.info("Condition set resulted in %s samples", len(samples))
        experiment_design = experiment_design.append(samples, ignore_index=True)

    return experiment_design.drop_duplicates()


def get_container_assignment(input):
    """
    Pick a container for each combination of batch factors to avoid search
    """
    containers = input['containers']
    batch_factors = { x : y for x, y in input['factors'].items() if y['ftype'] == 'batch' }
    ## Hack to dropna, really need to determine which groups are consistent and use that as a lower bound on
    ## the container set.  Using dropna assumes that rows with nan will subsume another row.
    batch_types = get_sample_types(batch_factors, input['requirements']).drop_duplicates().dropna() 
    l.debug("batch_types: %s", batch_types)

    assert(len(batch_types) <= len(containers))

    container_assignment = batch_types.reset_index(drop=True).to_dict('index')

    ## Set keys to str
    container_assignment = { str(list(containers.keys())[k]):v for k,v in container_assignment.items() }

    l.debug("container_assignment: %s", container_assignment)
    return container_assignment

def fill_empty_aliquots(factors, requirements, num_fill):
    if num_fill > 0:
        constraint = {
                    "factors": [
                        {
                            "factor": "strain",
                            "values": [
                                "None"
                            ]
                        },
                        {
                            "factor": "replicate",
                            "values": [i for i in range(1, num_fill+1)]
                        }
                    ]
                }
        new_requirements = requirements + [constraint]

        new_factors = factors.copy()
        if "None" not in new_factors['strain']['domain']:
            new_factors['strain']['domain'].append("None")
        new_factors['replicate']['domain'] = [1, max(factors['replicate']['domain'][1], num_fill)]

        l.info("new_factors: %s", new_factors['strain']['domain'])
        l.info("replicate_domain: %s", new_factors['replicate']['domain'])
        l.info("new_requirement: %s", constraint)
        return new_factors, new_requirements
    else:
        return factors, requirements

def solve1(input, pick_container_assignment=False, hand_coded_constraints=None):
    """
    Convert input to encoding and invoke solver.  Return model if exists.
    """
    pd.set_option("display.max_rows", 200)
    if pick_container_assignment:
        input['container_assignment'] = get_container_assignment(input)
    else:
        input['container_assignment'] = None

    containers = input['containers']
    l.debug(containers)
    strain_counts = { k : get_strain_count(v) for k, v in containers.items()}
    l.debug("container strain count: %s", strain_counts)
    container_strains = set([x for _, s in strain_counts.items() for x in s])
    l.debug("container_strains %s", container_strains)
    
    if not input['samples']:

        sample_factors = { x : y for x, y in input['factors'].items() if y['ftype'] == 'sample' }
        non_sample_factors = {x : y for x, y in input['factors'].items() if y['ftype'] != 'sample'}
        l.info("sample_factors: %s", sample_factors)

        ## Get the requirements for each sample in the experiment
        sample_types = get_sample_types(input['factors'], input['requirements'])
        l.debug("sample_types: %s", sample_types)
        requirement_strains = set(list(sample_types.strain.unique()))
        l.debug("requirement strains: %s", sample_types.strain.unique())

        l.debug("strains unique to containers: %s", container_strains.difference(requirement_strains))
        l.debug("strains unique to requirements: %s", requirement_strains.difference(container_strains))
        
        ## Get the number samples with identical sample factors 
        unique_samples = sample_types.pivot_table(index=list(sample_factors.keys()), aggfunc='size')
        #l.debug(unique_samples)

        ## Get the number of samples needed for each aliquot
        aliquot_samples =  sample_types.fillna(value=0).pivot_table(index=list(non_sample_factors.keys()), aggfunc='size', fill_value=1)
        l.debug("Required aliquots: %s", aliquot_samples)

        ## Num MediaControl wells in requirements
        num_media_control = len(sample_types.loc[sample_types.strain=="MediaControl"].drop(columns=list(sample_factors.keys())).drop_duplicates())

        ## Required batches
        batch_factors = [ x for x, y in input['factors'].items() if y['ftype'] == 'batch' ]
        batches = sample_types[batch_factors].drop_duplicates()
        l.debug("Required batches: %s", batches)


        ## Check how many aliquots are needed and ensure that containers will provide that many
        num_container_aliquots = sum([len(container['aliquots']) for container_id, container in containers.items()])
        num_empty_wells = sum([c["None"] for c_id, c in strain_counts.items() if "None" in c]) + sum([c[None] for c_id, c in strain_counts.items() if None in c])
        num_media_wells = sum([c["MediaControl"] for c_id, c in strain_counts.items() if "MediaControl" in c])
        num_empty_for_none = (num_empty_wells + num_media_wells) - num_media_control
        l.info("# container aliquots = %s, num_aliquots needed = %s, empty = %s", num_container_aliquots, len(aliquot_samples), num_empty_for_none)
        assert(num_container_aliquots >= len(aliquot_samples))
        
        ## Fill requirements with empty samples to place in unused aliquots
        num_unused_aliquots = num_container_aliquots -  num_empty_wells
        input['factors'], input['requirements'] = fill_empty_aliquots(input['factors'], input['requirements'], num_empty_for_none)
        
        ## Get the samples in common with all aliquots
        sample_groups = sample_types.groupby(list(non_sample_factors.keys()))
        common_samples = sample_types[list(sample_factors.keys())].drop_duplicates()#.reset_index()
        num_samples = aliquot_samples.max()
        l.info("num_samples: %s", num_samples)
        for g, df in sample_groups:
            common_samples = common_samples.merge(df[list(sample_factors.keys())], on=list(sample_factors.keys()), how="inner")#.reset_index()
        l.debug(common_samples)
        
        input['samples'] = {
            c: {
                a : {
                    x : "x{}_{}_{}".format(x, a, c) for x in range(0, num_samples) }
                for a in containers[c]['aliquots'] }
            for c in containers }

        input['sample_types'] = { i : x for i, x in enumerate(common_samples.to_dict('records'))}
        l.debug("sample_types: %s", input['sample_types'])
        for i in range(len(common_samples), num_samples):
            l.debug("Adding Free sample %s", i)
            input['sample_types'][i] = None

    l.info("Generating Constraints ...")
    variables, constraints = generate_constraints1(input)

    if hand_coded_constraints:
        for hc_c in hand_coded_constraints:
            l.info("hand coded constraint: %s", hc_c)
            constraints = And(eval(hc_c), constraints)
    
    l.info("Solving ...")
    model = get_model(constraints)

    if model is None:
        from pysmt.rewritings import conjunctive_partition
        conj = conjunctive_partition(constraints)
        ucore = get_unsat_core(conj)
        l.info("UNSAT-Core size '%d'" % len(ucore))
        for f in ucore:
            l.info(f.serialize())

    
    return model, variables



def get_model_pd(model, variables, factors):

    if not model:
        l.info("No Solution Found!")
        return None

    experiment_df = pd.DataFrame()
    batch_df = pd.DataFrame()
    column_df = pd.DataFrame()
    na_column_df = pd.DataFrame()
    row_df = pd.DataFrame()
    aliquot_df = pd.DataFrame()
    sample_df = pd.DataFrame()

    def info_to_df(info, value):
        def sub_factor_value(x, value):
            for col in x.index:
                if col in factors:
                    #if col == "temperature":
                    #l.debug("Set %s = %s", col, value)
                    if value.is_int_constant():
                        x[col] = int(value.constant_value())
                    elif value.is_real_constant():
                        x[col] = float(value.constant_value())
            return x

        
        info_df = pd.DataFrame()
        df = info_df.append(info, ignore_index=True)
        if value.is_int_constant() or value.is_real_constant():
            df = df.apply(lambda x: sub_factor_value(x, value), axis=1)            
        return df

    def merge_info_df(df, info, value, on):
        if len(df) > 0:
            #l.debug("Merge L: %s", df)
            #l.debug("Merge R: %s", info_to_df(info))
            #onon = df.columns.intersection(info_to_df(info).columns)
            #l.debug("onon %s", onon.values)
            #df = df.merge(info_to_df(info),  how='outer', on=list(onon.values))#, suffixes=('', '_y'))
            def preserve_na(s1, s2):
                for i, v in s2.items():
                    if v == "NA" or pd.isnull(s1[i]):
                        s1[i] = v
                return s1
            
            if on:
                df = df.set_index(on).combine_first(info_to_df(info, value).set_index(on)).reset_index()
                #df = df.set_index(on).combine(info_to_df(info, value).set_index(on), preserve_na, overwrite=False).reset_index()
            else:
                df = df.combine_first(info_to_df(info, value))
                #df = df.combine(info_to_df(info, value), preserve_na, overwrite=False)
            #to_drop = [x for x in df if x.endswith('_y')]
            #df = df.drop(to_drop, axis=1)

            #l.debug("Merge O: %s", df)
        else:
            df = df.append(info_to_df(info, value), ignore_index=True)
        return df

    for var, value in model:
        if value.is_true() or value.is_int_constant() or value.is_real_constant():
            if str(var) in variables['reverse_index']:
                #l.debug("{} = {}".format(var, value))
                info = variables['reverse_index'][str(var)]
                
               # l.debug("info = %s", info)
                if info['type'] == 'aliquot':
                    aliquot_df = merge_info_df(aliquot_df, info, value, ["aliquot", "container"])
                elif info['type'] == 'sample':
                    sample_df = merge_info_df(sample_df, info, value, "sample")
                elif info['type'] == 'batch':
                    batch_df = merge_info_df(batch_df, info, value, "container")
                    #print(var, value, value.is_int_constant())
                    #print(batch_df)
                elif info['type'] == 'column':
                    column_df = merge_info_df(column_df, info, value, "column")
                elif info['type'] == 'na_column':
                    na_column_df = merge_info_df(na_column_df, info, value, "column")

                elif info['type'] == 'row':
                    row_df = merge_info_df(row_df, info, value, "row")

                elif info['type'] == 'experiment':
                    l.debug("info: %s", info)
                    experiment_df = experiment_df(experiment_df, info, value, None)
            
    l.debug("aliquot_df %s", aliquot_df)
    l.debug("sample_df %s", sample_df)
    l.debug("column_df %s", column_df)
    l.debug("na_column_df %s", na_column_df)
    l.debug("row_df %s", row_df)
    l.debug("batch_df %s", batch_df)
    l.debug("experiment_df %s", experiment_df)
    df = aliquot_df.drop(columns=['type']).merge(sample_df.drop(columns=['type']), on=["aliquot", "container"])
    if len(column_df) > 0:
        if len(na_column_df) > 0:
            ## Override values chosen for columns by NA if needed
            column_df = na_column_df.set_index("column").combine_first(column_df.set_index("column")).reset_index()
        df = df.merge(column_df.drop(columns=['type']), on=["container", "column"])
    if len(row_df) > 0:
        df = df.merge(row_df.drop(columns=['type']), on=["container", "row"])

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
    return df

    
