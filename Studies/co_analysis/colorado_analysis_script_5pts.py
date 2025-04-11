import json
import pandas as pd
import os

import PySAM
import PySAM.Singleowner as single_owner
import PySAM.Pvsamv1 as pv_model
import PySAM.Grid as grid
import PySAM.Utilityrate5 as utility_rate
import PySAM.ResourceTools as tools

from functools import partial
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import multiprocessing as mp
import numpy as np
import seaborn as sns

def get_cmod_json(json_file_path):
    with open(json_file_path, 'r') as f:
        dict = json.load(f)
    return dict

#ppa_prices = [x / 1000.0 for x in range(0, 70, 2)] # Range only accepts integers. PPA prices of $0.0/kWh to $0.2/kWh in increments of $0.005
#crop_profit = range(0, 500000, 10000) # $5,000/acre * 100 acres in $100/acre increments

def run_analysis(i):
    locs = pd.read_excel("Colorado Counties Multiple Points.xlsx", header=1)

    equipment_data = pd.read_excel("equipment_width_and_crop_profit_factor.xlsx", header=1)
    equipment_data = equipment_data.set_index("Row spacing (ft)")
    crop_multipliers = equipment_data[['Onions - profit', 'Potatoes - profit', 'Sugar Beets - profit', 'Wheat - profit', 'Equipment agnostic - profit', 'Open air - profit']].copy()
    crop_multipliers = crop_multipliers.dropna()
    crops = ["Onions"]

    config = i[0]
    co_idx = i[1]

    acres = 160
    min_profit = -1000 * acres
    max_profit = 1000 * acres

    #min_profit = 0 * acres
    #max_profit = 700 * acres

    #ppa_prices =  [x / 2000.0 for x in range(0, 141, 1)]
    #ppa_prices = [0.045, 0.047, 0.05, 0.052]
    #ppa_prices =  [x / 2000.0 for x in range(88, 121, 1)] #0.044 to #0.06
    ppa_prices = [0.051]
    #crop_profit = range(min_profit, max_profit + 1, 50 * acres)
    crop_profit = [100 * acres]
    #crop_profit = [max_profit]

    full_data = []
    columns = ["County", "Lat", "Long", "Config", "Crop", "Solar capacity", "CAPEX", "PPA price", "Open Air Crop Profit", "Specific Crop Profit", \
                "NPV", "IRR", "Capacity Factor", "Annual Energy Production (year 1)", "Energy Value (year 1)", "Discounted Payback"]

    data_folder = "input_json/"
    output_folder = "pv_json/"

    pv = pv_model.default("FlatPlatePVSingleOwner")

    pv_json = get_cmod_json(data_folder + config + "_pvsamv1.json")
    so_json = get_cmod_json(data_folder + config + "_singleowner.json")

    co_name = locs.loc[co_idx]["COUNTY"]
    co_lat = locs.loc[co_idx ]["CENT_LAT"]
    co_long = locs.loc[co_idx ]["CENT_LONG"]

    filename = output_folder + co_name + "_" + str(config) + ".json"
    
    print(config, co_name)
    gen = []
    annual_energy = 0
    cf = 0
    if not os.path.isfile(filename):

        for k, v in pv_json.items():
            if "adjust_" in k:
                k = k.replace("adjust_", "")
            try:
                pv.value(k, v)
            except AttributeError:
                print("Error ", k)
        
        pv.SolarResource.solar_resource_file = "weather_data/nsrdb_" + str(co_lat) + "_" + str(co_long) +"_psm3-tmy_60_tmy.csv"

        pv.execute()
        gen = pv.Outputs.gen
        annual_energy = pv.Outputs.annual_energy
        cf = pv.Outputs.capacity_factor

        with open(filename, "w") as f:
            data = {"gen" : pv.Outputs.gen,
                    "cf" : pv.Outputs.capacity_factor,
                    "annual_energy" : pv.Outputs.annual_energy}
            json.dump(data, f, indent=4)

    else:
        with open(filename, "r") as f:
            data = json.load(f)
            gen = data["gen"]
            cf = data["cf"]
            annual_energy = data["annual_energy"]

    for price in ppa_prices:
        for profits in crop_profit:
            for crop in crops:
                so = single_owner.default("FlatPlatePVSingleOwner")

                for k, v in pv_json.items():
                    if "adjust_" in k:
                        k = k.replace("adjust_", "")
                    try:
                        pv.value(k, v)
                    except:
                        pass 

                for k, v in so_json.items():
                    try:
                        so.value(k, v)
                    except AttributeError:
                        print("Error ", k)

                so.SystemOutput.gen = gen
                so.SystemOutput.system_pre_curtailment_kwac = gen
                so.FinancialParameters.system_capacity = pv.SystemDesign.system_capacity

                so.Revenue.ppa_price_input = [price]
                
                row_float = float(config[0:4])
                multiplier = crop_multipliers.loc[row_float][crop + " - profit"]
                specific_profit = profits * multiplier
                so.SystemCosts.om_fixed = [-1.0 * specific_profit]

                so.execute()

                # ["County", "Lat", "Long", "Config", "Crop", "Solar capacity", "CAPEX", "PPA price", "Open Air Crop Profit", "Specific Crop Profit" \
                # "NPV", "IRR", "Capacity Factor", "Annual Energy Production (year 1)", "Energy Value (year 1)", "Discounted Payback"]

                npvs = so.Outputs.cf_project_return_aftertax_npv
                payback = 99
                for i, npv in enumerate(npvs):
                    if npv > 0:
                        payback = i
                        break

                run_data = [co_name, # County
                            co_lat, # Lat
                            co_long, # Long
                            config, # Config
                            crop, # Crop
                            so.SystemOutput.system_capacity, # Solar capacity
                            so.SystemCosts.total_installed_cost + so.FinancialParameters.construction_financing_cost, # CAPEX
                            price, # PPA price
                            profits, # Open Air Crop Profit
                            specific_profit, # Specific Crop Profit
                            so.Outputs.project_return_aftertax_npv, #NPV
                            so.Outputs.project_return_aftertax_irr, #IRR
                            cf, # Capacity Factor - DC
                            annual_energy, # Annual Energy
                            list(so.Outputs.cf_energy_value)[1], # Energy Value
                            payback # Discounted Payback
                            ]

                full_data.append(run_data)

    df = pd.DataFrame(data=full_data, columns=columns)
    return df


if __name__ == "__main__":
    cores = 13
    #configs = ['16.7ft', '21.7ft', '26.7ft', '31.7ft', '36.7ft', '41.7ft', '46.7ft', '51.7ft', '56.7ft', '61.7ft', '66.7ft', '71.7ft', '76.7ft']
    configs = ['31.7ft']
    counties = range(0,320)
    jobs = itertools.product(configs, counties)

    print(jobs)

    with mp.Pool(cores) as pool:
        results = pool.map(partial(run_analysis), jobs)
        results_df = pd.concat(results)

    results_df.to_csv("single_econ_multi_county_multi_point.csv")