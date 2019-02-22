import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def get_per_experiment_od_plot(experiment_groups_result):
    ax=plt.axes()
    ax.clear()
    ax.errorbar(experiment_groups_result['pre_OD600']['mean'], 
                experiment_groups_result['OD600']['mean'], 
                experiment_groups_result['pre_OD600','std'], 
                experiment_groups_result['OD600','std'],  
                alpha=.7,
                linestyle='None')
    ax.set_xlabel('Pre OD')
    ax.set_ylabel('Post OD')
    ax.set_ylim(0,4)
    ax.set_xlim(0,4)
    ax.set_aspect('equal', 'box')
    ax.set_title('Pre/Post OD by Experiment' )
    return ax

def get_per_experiment_od_by_od_plot(experiment_od_groups_result):
    f, axarr = plt.subplots(4,1,                         
                        figsize=(60, 30))
    ods=experiment_od_groups_result.od.unique()
    #f.subplots_adjust(wspace=2)
    for j, od in enumerate(ods):
        #my_df = df.loc[(df['strain_circuit'] == circuit] #.loc[result['strain'] == 'UWBF_NAND_01']
        m_my_df = experiment_od_groups_result.loc[(experiment_od_groups_result['od'] == od)] 


        axarr[j].errorbar(m_my_df['pre_OD600']['mean'], m_my_df['OD600']['mean'], m_my_df['pre_OD600','std'], m_my_df['OD600','std'],  alpha=.7,linestyle='None')

        axarr[j].set_xlabel('Pre OD')
        axarr[j].set_ylabel('Post OD')
        #axarr[j, i].set_xscale("log", nonposx='clip')
        #axarr[j, i].set_yscale("log", nonposy='clip')
        axarr[j].set_ylim(0,4)
        axarr[j].set_xlim(0,4)
        axarr[j].set_aspect('equal', 'box')
        axarr[j].set_title('Pre/Post OD per Experiment, OD = ' + str(od)  )
        #axarr[j].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    return f

def get_strain_statistics_by_od_plot(result):
    circuits = ['AND', 'OR', 'NAND', 'NOR', 'XOR', 'XNOR']
    ods = result.od.unique()
    ods.sort()
    f, axarr = plt.subplots(4,6,                         
                        figsize=(60, 30))
    for j, od in enumerate(ods):
        for i, circuit in enumerate(circuits):
            m_my_df = result.loc[(result['strain_circuit'] == circuit) & \
                                 (result['od'] == od)] 

            colored_col = 'strain'
            colored_vals = np.sort(m_my_df[colored_col].dropna().unique())
            colors = cm.rainbow(np.linspace(0, 1, len(colored_vals)))
            colordict = dict(zip(colored_vals, colors))  
            m_my_df.loc[:,"Color"] = m_my_df[colored_col].apply(lambda x: colordict[x])

            for strain in np.sort(m_my_df.strain.unique()):
                m_strain_df = m_my_df.loc[m_my_df['strain'] == strain]
                axarr[j, i].errorbar(m_strain_df['pre_OD600']['mean'],
                                     m_strain_df['OD600']['mean'],
                                     m_strain_df['pre_OD600','std'],
                                     m_strain_df['OD600','std'],  
                                     alpha=.7, 
                                     label=m_strain_df['strain'].values)

            axarr[j, i].set_xlabel('Pre OD')
            axarr[j, i].set_ylabel('Post OD')
            #axarr[j, i].set_xscale("log", nonposx='clip')
            #axarr[j, i].set_yscale("log", nonposy='clip')
            axarr[j, i].set_ylim(0,3)
            axarr[j, i].set_xlim(0,3)
            axarr[j, i].set_aspect('equal', 'box')
            axarr[j, i].set_title('OD = ' + str(od) + " " + circuit )
            axarr[j, i].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    return f

def get_strain_by_od_plot(df, result):

    circuits = ['AND', 'OR', 'NAND', 'NOR', 'XOR', 'XNOR']
    f, axarr = plt.subplots(1,6,                         
                        figsize=(100, 10))
    f.subplots_adjust(hspace=0.45, wspace=.25)
    for i, circuit in enumerate(circuits):
        my_df = df.loc[df['strain_circuit'] == circuit] #.loc[result['strain'] == 'UWBF_NAND_01']
        m_my_df = result.loc[result['strain_circuit'] == circuit] #.loc[result['strain'] == 'UWBF_NAND_01']


        colored_col = 'strain'
        colored_vals = np.sort(my_df[colored_col].dropna().unique())
        colors = cm.rainbow(np.linspace(0, 1, len(colored_vals)))
        colordict = dict(zip(colored_vals, colors))  

        my_df.loc[:,"Color"] = my_df[colored_col].apply(lambda x: colordict[x])
        m_my_df.loc[:,"Color"] = m_my_df[colored_col].apply(lambda x: colordict[x])




        for strain in np.sort(my_df.strain.unique()):
            strain_df = my_df.loc[my_df['strain'] == strain]
            m_strain_df = m_my_df.loc[m_my_df['strain'] == strain]
            axarr[i].scatter(strain_df['OD600'], strain_df['od'], label=strain, color=strain_df.Color)
            axarr[i].scatter(m_strain_df['OD600']['mean'], 
                             m_strain_df['od'], 
                             s=m_strain_df['OD600']['std']*10000, 
                             alpha=.7, 
                             label=None, 
                             color=m_strain_df.Color)

        axarr[i].set_xlabel('Culture OD')
        axarr[i].set_ylabel('Innoculation OD')
        axarr[i].set_xscale("log", nonposx='clip')
        axarr[i].set_yscale("log", nonposy='clip')
        axarr[i].set_ylim(.00001, .1)
        axarr[i].set_title('Mean Corrected OD by Strain, ' + circuit)
        axarr[i].legend()
        
    return f
        
def get_pre_post_od_by_target_od(df):
    ax = plt.axes()

    colored_vals = np.sort(df['od'].dropna().unique())
    colors = cm.rainbow(np.linspace(0, 1, len(colored_vals)))
    colordict = dict(zip(colored_vals, colors))  

    df["Color"] = df['od'].apply(lambda x: colordict[x])

    ods = df.od.unique()
    for od in ods:
        ax.scatter(df['pre_OD600'], df['OD600'], label=str(od), c=df['Color'], alpha=0.5)
    plt.ylabel("Post OD")
    plt.xlabel("Pre OD")
    plt.title("Pre vs Post OD by Innoculation OD")
    plt.legend()
    legend = ax.get_legend()
    legend.legendHandles[0].set_color(colordict[0.0003])
    legend.legendHandles[1].set_color(colordict[7.5e-05])
    legend.legendHandles[2].set_color(colordict[0.00015])
    legend.legendHandles[3].set_color(colordict[0.00075])

    return ax

