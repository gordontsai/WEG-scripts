import matplotlib
import numpy as np


###Feed in dictionary of model distributions and the model number you want, spits out the model
def selectModel(model_dict,model_number): #model_number goes from 1-5, 1 being best, 5 being worst
    name = []
    model_function = []
    for k,v in model_dict.items():
        name.append(k)
        model_function.append(v)

    model = model_function[model_number]

    return model


def avgMPH(annualMiles):
    if annualMiles/365>100:
        mph = 60
    elif annualMiles/365 >50:
        mph = 40
    elif annualMiles/365 > 30:
        mph = 35
    elif annualMiles/365 > 20:
        mph = 30
    else:
        mph = 25
    return mph


def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'


def get_decision_percent(Decision):
  Uber_wins = 0
  Car_wins = 0

  for i in range(0,len(Decision)):
    if Decision[i] < 0:
      Car_wins += 1
    else:
      Uber_wins += 1

  Uber_percent = float(Uber_wins)/len(Decision)
  Car_percent = float(Car_wins)/len(Decision)

  print(Uber_percent, 'of the US is better off Ubering')
  print(Car_percent, 'of the US is better off Car Owning')

  return {'uber':Uber_percent,'car':Car_percent}
