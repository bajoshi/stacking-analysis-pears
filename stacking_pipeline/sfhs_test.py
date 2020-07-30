import numpy as np

def plot_sfhs():

    num_models = 10000
    tform = np.linspace(13.5, 1.5, num_models)

    a_list = 

    for i in range(20): 
        gamma = np.random.random() # in Gyr 
        current_tform = np.random.choice(tform) 
         
        tstep = 1e5 
        t = np.arange(current_tform * 1e9, 0+tstep, -1*tstep)  # in years 
        t /= 1e9  # converted back to Gyr 
    
        print("Formation time [yr]:", t[0]) 
        print("Gamma:", gamma)
         
        # Find parameter A 
        A = np.random.choice(a_list)

        print("A:", A)
    
        # Normalization of SFR assuming total mass generated is one solar 
        sfr0 = gamma / (np.exp(-1 * current_tform) - 1.0)  # Now this should have the correct units of solar masses per year 
        print("SFR normalization [M_sol/yr]:", sfr0)
    
        sfr = np.exp(-1 * gamma * t) 
        print("Starting and ending SFRs resp. [M_sol/yr]:", sfr[0], sfr[-1])
    
        total_mass_gen = trapz(sfr, t) 
        print("Total mass generated [M_sol; should be close to 1.0]:", total_mass_gen)
    
        fig = plt.figure() 
        ax = fig.add_subplot(111)
        ax.plot(t, sfr) 
        plt.show() 
        plt.clf() 
        plt.cla() 
        plt.close() 
    
        break 

    return None

def main():

    plot_sfhs()

    return None

if __name__ == '__main__':
    main()
    sys.exit(0)