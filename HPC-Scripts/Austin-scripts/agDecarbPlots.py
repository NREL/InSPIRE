# -*- coding: utf-8 -*-
"""
Ag Decarb Plots
Created on Thu Jul 20 16:08:10 2023

@author: akinzer
"""
# In[0]:
import os
import pandas as pd
import numpy as np
import plotly as plt
import matplotlib
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot

os.chdir('C:\\Users\\akinzer\\Documents')

# In[1]:
df1 = pd.read_csv('FarmFuelExpenses.csv')

fig = px.area(df1, x="Year",y="Fuel Expenditure",color="Fuel Type", line_group="Fuel Type", 
              pattern_shape="Fuel Type")
fig.update_layout(margin_pad=10, yaxis_title="Fuel Expenditure (Billions of US$)")
fig.update_layout(yaxis=dict(tickfont=dict(size=16), titlefont=dict(size=20)), 
                  xaxis=dict(tickfont=dict(size=16), titlefont=dict(size=20)),
                  legend=dict(font=dict(size=20)))



plot(fig)

# In[2]:

#df2 = px.data.gapminder()