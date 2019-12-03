import os
import json

from pysd2cat.analysis.plate_layout import solve, solve1, get_model_pd

def test_toy_problem():
    """
    Small problem with two aliquots.
    """
    ## Symbols denoting the samples
    samples = ["x1", "x2", "x3", "x4"]

    ## Factor definitions
    factors = { 
        "media" : {
            "domain" : ['m1', "m2"],
            "ftype" : "batch"
        }, 
        "strain" : {
            "domain" : [ "s1", "s2"],
            "ftype" : "sample"
        },
        "temperature" : {
            "domain" : [ "30", "35"],
            "ftype" : "experiment"
        },
        "measurement_type" : {
            "domain" : [ "PLATE_READER", "FLOW"],
            "ftype" : "shadow"
        },
        "inducer" : {
            "domain" : [ "0", "1"],
            "ftype" : "column"
        }
    }

    containers = { "c1" : {
                        "aliquots" : {
                            "a1" : {
                                "strain" : "s1"
                            },
                            "a2" : {
                                "strain" : "s2"
                            } 
                        },
                        "columns" : { "col1" : ["a1", "a2"] }
                    } 
                 }



    ## Requirement for the samples
    requirements = [
        {
            "factors" : [
                {
                    "factor" : "media", "values" : [ "m2" ]
                },
                {
                    "factor" : "strain", "values" : [ "s1", "s2" ]
                },
                {
                    "factor" : "temperature", "values" : [ "30" ]
                },
                {
                    "factor" : "measurement_type", "values" : [ "PLATE_READER", "FLOW" ]
                },
                {
                    "factor" : "inducer", "values" : [  "1" ]
                }
            ]
        }
    ]

    check(samples, factors, containers, requirements)

def check(samples, factors, containers, requirements):
    inputs = {
        "samples" : samples,
        "factors" : factors,
        "containers" : containers,
        "requirements" : requirements
    }
    check_inputs(inputs)
    
def check_inputs(inputs, method='solve'):
    model, variables = eval(method)(inputs)
    if model:
        print(model)
    else:
        print("No solution found")
    assert(model)

    get_model_pd(model, variables)
    
def test_growth_curve():
    """
    Scenario from Y4D growth curves
    """
    scenario_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../resources/smt/y4d_gc_smt_inputs.json')
    with open(scenario_file, 'r') as f:
        inputs = json.load(f)
        check_inputs(inputs)
    
    pass

def test_growth_curve1():
    """
    Scenario from Y4D growth curves
    """
    scenario_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../resources/smt/y4d_gc_smt_inputs.json')
    with open(scenario_file, 'r') as f:
        inputs = json.load(f)
        check_inputs(inputs, method='solve1')
    
    pass
