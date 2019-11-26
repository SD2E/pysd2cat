from pysmt.shortcuts import Symbol, And, Or, Not, Implies, Equals, Iff, is_sat, get_model, GT, GE, LT, Int, String
from pysmt.typing import INT, StringType

import logging


l = logging.getLogger(__file__)
l.setLevel(logging.DEBUG)



def generate_variables(inputs):
    """
    Encoding variables and values
    """

    
    samples = inputs['samples']
    factors = inputs['factors']
    containers = inputs['containers']
    aliquots = [a for c in containers for a in containers[c]['aliquots']]

    variables = {}
    variables['tau_symbols'] = { x: Symbol("tau_{}".format(x), INT) for x in samples }
    variables['sample_factors'] = \
      { 
        sample: {
            factor : Symbol("{}({})".format(factor, sample), INT) for factor in factors 
        }  for sample in samples
      }
    values = {}
    values['perp'] = Int(-1)
    values['min_aliquot'] = Int(0)
    values['max_aliquot'] = Int(len(aliquots))

    variables['exp_factor'] = \
      {
          factor : Symbol("{}_exp".format(factor), INT) 
            for factor in factors if factors[factor]['ftype'] == "experiment"
      }
    variables['batch_factor'] = \
      {
          factor : { 
                  container : Symbol("{}_{}_batch".format(factor, container), INT)  
                  for container in containers
            }
          for factor in factors if factors[factor]['ftype'] == "batch"
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


    bounds = generate_bounds(inputs, variables, values)
    
    samples = inputs['samples']
    factors = inputs['factors']
    containers = inputs['containers']
    requirements = inputs['requirements']

    aliquots = [a for c in containers for a in containers[c]['aliquots']]


    tau_symbols = variables['tau_symbols']
    sample_factors = variables['sample_factors']
    exp_factor = variables['exp_factor']
    batch_factor = variables['batch_factor']

    perp = values['perp']
    
    # (1), Each aliquot has a sample mapped to it
    aliquot_sample_constraint = \
      And([
        Or([Equals(tau_symbols[x], Int(a))
            for x in tau_symbols ])
        for a in range(0, len(aliquots))
        ])
    l.debug("aliquot_sample_constraint: %s", aliquot_sample_constraint)
    # (2)
    mapped_are_assigned_constraint = And([ Iff(Equals(tau_symbols[x], perp), 
                                           And([ Equals(sample_factors[x][f], perp) for f in factors ])) 
                                     for x in samples])
    l.debug("mapped_are_assigned_constraint: %s", mapped_are_assigned_constraint)
        
    # (3)
    uniformly_assigned_factors_constraint = And([ Implies(Equals(sample_factors[x][f], perp), 
                                                      Equals(sample_factors[x][fp], perp))
                                              for f in factors 
                                              for fp in factors
                                              for x in samples])
    l.debug("uniformly_assigned_factors_constraint: %s", uniformly_assigned_factors_constraint)
    # (4)
    requirements_constraint = \
    And([Implies(Not(Equals(tau_symbols[x], perp)),
        Or([
            And([
                Or([Equals(sample_factors[x][f['factor']], 
                           Int(factors[f['factor']]["domain"].index(level)))
                    for level in f['values']]) 
                for f in r["factors"]]) 
            for r in requirements]))
        for x in samples])
    l.debug("requirements_constraint: %s", requirements_constraint)

    # (5)
    aliquot_properties_constraint = \
      And([Implies(Equals(tau_symbols[x], Int(aliquots.index(aliquot))),
                   And([Equals(sample_factors[x][factor], 
                               Int(factors[factor]["domain"].index(level)))
                        for factor, level in aliquot_properties.items()]))
               for x in samples
               for _, c in containers.items()
               for aliquot, aliquot_properties in c['aliquots'].items()
            ])
    l.debug("aliquot_properties_constraint: %s", aliquot_properties_constraint)

    # (6)
    experiment_factors_constraint = \
    And([Implies(Not(Equals(tau_symbols[x], perp)),
              Equals(sample_factors[x][factor], exp_factor[factor]))
         for factor in factors if factors[factor]["ftype"] == "experiment"
         for x in samples])
    l.debug("experiment_factors_constraint: %s", experiment_factors_constraint)

    # (7)
    batch_factors_constraint = \
      And([ 
        Implies(Equals(tau_symbols[x], Int(aliquots.index(aliquot))), 
                And([Equals(sample_factors[x][factor], batch_factor[factor][container_id])
                     for factor in factors if factors[factor]["ftype"] == "batch"]))    
        for container_id, container in containers.items()
        for aliquot, aliquot_properties in container['aliquots'].items()
        for x in samples
        ])
    l.debug("batch_factors_constraint: %s", batch_factors_constraint)

    # (8)
    sample_factors_constraint = \
    And([
        And([
            Implies(Equals(tau_symbols[x], tau_symbols[xp]),
                    And([
                        Equals(sample_factors[x][factor], sample_factors[xp][factor])
                        for factor in factors if factors[factor]["ftype"] == "batch"]))
            for xp in samples])
        for x in samples])
    l.debug("sample_factors_constraint: %s", sample_factors_constraint)

    # (9)
    column_factors_constraint = \
    And([ 
        Implies(And(Equals(tau_symbols[x], Int(aliquots.index(a1))),
                    Equals(tau_symbols[xp], Int(aliquots.index(a2)))),
                And([Equals(sample_factors[x][factor], sample_factors[xp][factor])
                     for factor in factors if factors[factor]["ftype"] == "column"]))    
        for xp in samples
        for x in samples           
        for container_id, container in containers.items()
        for column_id, column in container['columns'].items()
        for a1 in column
        for a2 in column
        ])
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
                And([Equals(sample_factors[x][factor], 
                           Int(factors[factor]["domain"].index(level)))
                    for factor, level in xr.items()]) 
                for x in samples]) 
            for xr in expand_requirement(r)])
        for r in requirements])
    l.debug("satisfy_every_requirement: %s", satisfy_every_requirement)

#print(sample_factors_constraint)

    f = And(
    bounds,
    aliquot_sample_constraint, # 1
    mapped_are_assigned_constraint, #2
    uniformly_assigned_factors_constraint, #3
    requirements_constraint, #4
    aliquot_properties_constraint, #5
    experiment_factors_constraint, #6
    batch_factors_constraint, #7
    sample_factors_constraint, #8
    column_factors_constraint, #9   
    satisfy_every_requirement #10
    )
    l.debug("Constraints: %s", f)

    return f

def solve(input):
    """
    Convert input to encoding and invoke solver.  Return model if exists.
    """

    if not input['samples']:
        input['samples'] = [ "x{}".format(x) for x in range(0, 84) ]
    
    constraints = generate_constraints(input)
    model = get_model(constraints)
    return model
