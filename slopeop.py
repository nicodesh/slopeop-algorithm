import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Slope OP Algorithm
def slopeop(data, vmin=0, vmax=100, beta=500):
    """ Compute breakpoints from your data.

    Args:
        data (Numpy array or Pandas series): your data
        vmin (int): floor state (default 0)
        vmax (int): ceiling state (default 100)
        beta (int): cost penalty (default 500)

    Returns:
        cp (Numpy array): The changepoints vector.
        U (Numpy array): The vertical values vector.
        Qn (Numpy array): The costs of a last point, for each possible state.
        Q (Numpy array): The costs matrix.
        beta (int): THe cost penalty as specified by the user (default 500)

    """

    # Params
    y = data.copy()

    # Initialization
    n = len(y)
    n_rows = vmax-vmin+1
    
    Q = np.zeros(shape=(vmax+1, n+1))
    Q_canal = [(vmin,vmax)]
    cp = np.zeros(shape=(vmax+1, n)) -1
    U = np.zeros(shape=(vmax+1, n)) -1
    
    # Pre-processing
    
    # Cumsum of y
    st_one = np.cumsum(data)
    
    # Cumsum of y^2
    st_two = np.cumsum(np.array(data)**2)
    
    # Cumsum of i*yi
    st_plus = np.cumsum(np.array(data)*np.arange(len(data)))

    # Cost Function
    def C(tauplusone, t, a, b):
        """ Return the cost."""
        tau = tauplusone-1
        result = (st_two[t]
        - st_two[tau]
        + ((b**2 - a**2)/2)
        + (((a**2 + a*b + b**2)/3)*(t - tau))
        + (((b-a)**2) / (6*(t-tau)))
        - ((2/(t-tau)) * (((t*a) - (b*tau)) * (st_one[t] - st_one[tau]) + ((b-a) * (st_plus[t] - st_plus[tau])))))
        
        return result

    # First cost
    for v in range(vmin, vmax+1):
        Q[v][0] = (y[0] - v)**2

    # Parse the series
    for t in range(1, n):

        # Parse the vertical
        for v in range(vmin, vmax+1):

            Q_temp = np.inf
            argmin_tau = 0
            argmin_u = 0

            # Parse from the beginning - horizontal
            
            for tau in range(0,t):
                
                # Channel - Born with v hat if necessary
                # Compute v hat
                try:
                    v_hat = ((6/((t-tau-1)*(2*(t-tau)-1)))
                            * (t * (st_one[t] - st_one[tau]) - (st_plus[t] - st_plus[tau]))
                            - (v * ((t-tau+1) / (2*(t-tau)-1))))
                except:
                    v_hat = (Q_canal[tau][0] + Q_canal[tau][1]) / 2
                
                # Round to the nearest integer
                pos_v_hat = int(round(v_hat))
                
                # Keep it in the limits
                if (pos_v_hat > vmax):
                    pos_v_hat = vmax
                elif (pos_v_hat < vmin):
                    pos_v_hat = vmin
                
                # Restrain the canal
                vmin_temp = min(pos_v_hat, Q_canal[tau][0])
                vmax_temp = max(pos_v_hat, Q_canal[tau][1])
                
                # Parse from the begenning - vertical
                for u in range(vmin_temp, vmax_temp+1):
                
                    # Compute the cost
                    current_val = Q[u][tau] + C(tau+1,t,u,v) + beta 

                    if current_val < Q_temp:
                        Q_temp = current_val
                        argmin_tau = tau
                        argmin_u = u

            Q[v,t] = Q_temp
            cp[v,t] = argmin_tau
            U[v,t] = argmin_u
        
        # Channel - Compute interval of Q
        q_left = vmin
        q_right = vmax
        
        for i in range(vmin, vmax):
            if Q[i,t] > Q[i+1,t]:
                q_left = i+1
            else:
                break
                
        for i in range(vmax, vmin, -1):
            if Q[i,t] > Q[i-1,t]:
                q_right = i-1
            else:
                break
                
        Q_canal.append((q_left, q_right))
                
    return cp, U, Q[:,-2], Q, beta

# Backtracking
def slopeop_bt(data, vmin, vmax, cp, U, Qn):
    """ Compute the backtracking from a SlopeOP vector.

    Args:
        data (Numpy array or Pandas series): The original series
        vmin (int): Floor state
        vmax (int): Ceiling state
        cp (Numpy array): Changepoints vector from the SlopeOP algorithm
        U (Numpy array): Vertical values corresponding to the changepoints
        Qn (Numpy array): Costs of the last horizontal point

    Returns:
        segments (list): Horizontal changepoints
        bpm (list): Correspoding vertical values for each changepoint

    """
    
    y = data.copy()
    
    # First BPM (based on optimal total cost)
    # The first BPM is actually the last one
    Qmin_val = min(Qn[vmin:vmax+1])
    Qmin_arg = np.where(Qn == Qmin_val)[0][0]
    
    # Initialization
    n = len(y)-1
    segments = [n]
    bpm = [Qmin_arg]
    tau = n
    print(Qmin_val)
    
    while tau > 0:
        segments.append(int(cp[bpm[-1]][int(tau)]))
        bpm.append(int(U[bpm[-1]][int(tau)]))
        tau = cp[bpm[-2]][int(tau)]
        
    segments.reverse()
    bpm.reverse()
        
    return segments, bpm

def merge_segments(data, segments, bpm):
    """ Merge the original dataset with segment (x) and values (y).

    Args:
        data (Numpy array or Pandas series): The original series
        segments (list): Segments as provided by the backtracking algorithm
        bpm (list): BPM as provided by the backtracking algorith

    Returns:
        df (Pandas dataframe): A dataframe with the original data plus the changepoints.

    """
    
    df = pd.DataFrame()
    df['index'] = np.arange(len(data))

    df_temp = pd.DataFrame()
    df_temp['index'] = pd.Series(segments).astype(int)
    df_temp['bpm'] = pd.Series(bpm)

    df = df.merge(df_temp, how="left", on="index")
    df.drop("index", axis=1, inplace=True)
    
    return df