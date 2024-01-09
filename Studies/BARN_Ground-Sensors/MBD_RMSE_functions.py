# look at residuals (MBD, RMSE) based on Grear, Gpoa and Gtotal_modeled. From B. Marion Solar Energy 2016
def MBD(meas,model):
    # MBD=100∙[((1⁄(m)∙∑〖(y_i-x_i)]÷[(1⁄(m)∙∑〖x_i]〗)〗)
    import pandas as pd
    df = pd.DataFrame({'model':model,'meas':meas})
    # rudimentary filtering of modeled irradiance
    df = df.dropna()
    minirr = meas.min()
    df = df[df.model>minirr]
    m = df.__len__()
    out = 100*((1/m)*sum(df.model-df.meas))/df.meas.mean()
    return out

def RMSE(meas,model):
    #RMSD=100∙〖[(1⁄(m)∙∑▒(y_i-x_i )^2 )]〗^(1⁄2)÷[(1⁄(m)∙∑▒〖x_i]〗)
    import numpy as np
    import pandas as pd
    df = pd.DataFrame({'model':model,'meas':meas})
    df = df.dropna()
    minirr = meas.min()
    df = df[df.model>minirr]
    m = df.__len__()
    out = 100*np.sqrt(1/m*sum((df.model-df.meas)**2))/df.meas.mean()
    return out

# residuals absolute output (not %) 
def MBD_abs(meas,model):
    # MBD=100∙[((1⁄(m)∙∑〖(y_i-x_i)]÷[(1⁄(m)∙∑〖x_i]〗)〗)
    import pandas as pd
    df = pd.DataFrame({'model':model,'meas':meas})
    # rudimentary filtering of modeled irradiance
    df = df.dropna()
    minirr = meas.min()
    df = df[df.model>minirr]
    m = df.__len__()
    out = ((1/m)*sum(df.model-df.meas))
    return out

def RMSE_abs(meas,model):
    #RMSD=100∙〖[(1⁄(m)∙∑▒(y_i-x_i )^2 )]〗^(1⁄2)÷[(1⁄(m)∙∑▒〖x_i]〗)
    import numpy as np
    import pandas as pd
    df = pd.DataFrame({'model':model,'meas':meas})
    df = df.dropna()
    minirr = meas.min()
    df = df[df.model>minirr]
    m = df.__len__()
    out = np.sqrt(1/m*sum((df.model-df.meas)**2))
    return out