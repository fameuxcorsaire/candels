###############################################################################
##################SCRIPT QUE FILTRA Y PROCESA DATOS DE CANDELS#################
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns 
from sklearn.linear_model import TheilSenRegressor
from astropy.cosmology import WMAP9
from astropy.coordinates import Distance
############################DEFINO FUNCIONES ÚTILES############################



def cuenta_rango(li, min, max): #Numero de elementos en rango de una lista
	ctr = 0
	for x in li:
		if min <= float(x) <= max:
			ctr += 1
	return ctr

def extrae_rango(li_ini,col,li_fin,min,max): #Extrae filas de lista según valores de col y las lleva a otra 
  #(debe tener previamente las dimensiones correctas)
    ctr=0
    for i in range(len(li_ini[:,col])):
        if min <= float(li_ini[i,col]) <= max:
            li_fin[ctr,:]=li_ini[i,:]
            ctr += 1
            
def round_to_n(x,n): #redondea a n cifras significativas
    for i in range(len(x)):
        if x[i]!=0.0:
            x[i]=round(x[i],n-1-int(np.floor(np.log10(abs(x[i])))))
        else:
            x[i]=0
    return x
        
            
def binear(data,n,decimal): #Crea n intervalos para datos equiespaciados
    return round_to_n(np.linspace(min(data),max(data),n),decimal)
    
    
    
            
########################CARGO DATOS Y PROCESADO INICIAL########################
            



            
##################################################
#SELECCIONA GALAXIAS CON Z ADECUADO Y CARGA DATOS#
##################################################

      
datos_z=np.loadtxt('redshift.txt', usecols = (0,1,2,4,5,6,7) ,skiprows=1, ) #Lo cargo como str, hay que converir a float
selec_z=np.zeros((cuenta_rango(datos_z[:,6].tolist(), 0, 0.5),7)) #Creo objeto para alacenar datos filtrados
extrae_rango(datos_z,6,selec_z,0,0.5)
msg='Dentro del rango hay '+ str(len(selec_z))+' galaxias'
print(msg)


#CARGO Z COMO DATASET DE PANDAS Y DIBUJO DATOS
#Creo el dataframe de z para z#
ds_z = pd.DataFrame({'id': selec_z[:, 0], 'zspec': selec_z[:, 1], 
                     'zspecflag': selec_z[:, 2],'ztier': selec_z[:, 3],'ztier_err': selec_z[:, 4],
                     'ztier_class': selec_z[:, 5],'zbest': selec_z[:, 6],})



#INCORPORA más DATOS para mi rango de Z#

datos_fast=np.loadtxt('pobestelar_fast.txt', usecols = (0,1,2,3,4,6,7,9,11,12,13) ,skiprows=1, ) #Lo cargo como str, hay que converir a float
selec_fast=np.zeros((cuenta_rango(datos_fast[:,1].tolist(), 0, 0.5),11)) #Creo objeto para alacenar datos filtrados
extrae_rango(datos_fast,1,selec_fast,0,0.5)

ds_fast = pd.DataFrame({'id': datos_fast[:, 0], 'zbest': datos_fast[:, 1],    #datos de galaxias hasta z=0.5
                     'ltau': datos_fast[:, 2],'metal': datos_fast[:, 3],'lage': datos_fast[:, 4],
                     'lmass': datos_fast[:, 5],'lsfr': datos_fast[:, 6],'la2t': datos_fast[:, 7],
                     'Mu': datos_fast[:, 8],'Mv': datos_fast[:, 9],'Mj': datos_fast[:, 10]})

fig_0, ((ax1,ax2)) = plt.subplots(1, 2,figsize=(12,4))

plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels

df = pd.DataFrame({'z':datos_fast[:,1]})
dat = df.z
z_plot=dat.hist(grid=0,ax=ax1,facecolor='silver',edgecolor='black', alpha=0.7,bins=binear(dat,12,1))
z_plot.set_xlabel('$Redshift$')
z_plot.set_ylabel('$N$')
z_describe=dat.describe() #Hago la estadística para el dataset de Z

z_plot.axvline(z_describe[1],color='lime',alpha=0.7)
z_plot.axvline(z_describe[4],color='magenta',alpha=0.4)
z_plot.axvline(z_describe[5],color='magenta',alpha=0.4)
z_plot.axvline(z_describe[6],color='magenta',alpha=0.4)

dat=ds_z['zbest']
z_plot=dat.hist(grid=0,ax=ax2,facecolor='silver',edgecolor='black', alpha=0.7,bins=binear(dat,6,1))
z_plot.set_xlabel('$Redshift$')
z_plot.set_ylabel('$N$')
z_describe=dat.describe() #Hago la estadística para el dataset de Z

z_plot.axvline(z_describe[1],color='lime',alpha=0.7)
z_plot.axvline(z_describe[4],color='magenta',alpha=0.4)
z_plot.axvline(z_describe[5],color='magenta',alpha=0.4)
z_plot.axvline(z_describe[6],color='magenta',alpha=0.4)

plt.savefig('figuras/z')
#%%
fig_0, ((ax3,ax4)) = plt.subplots(1, 2,figsize=(12,4))


df = pd.DataFrame({'z':datos_fast[:,1]})
dat = df.z[df.z<0.7]
z_plot=dat.hist(grid=0,ax=ax3,facecolor='silver',edgecolor='black', alpha=0.7,bins=binear(dat,12,1))
z_plot.set_xlabel('$Redshift$')
z_plot.set_ylabel('$N$')
z_plot.set_title('GOODS-N')
z_describe=dat.describe() #Hago la estadística para el dataset de Z

z_plot.axvline(z_describe[1],color='lime',alpha=0.7)
z_plot.axvline(z_describe[4],color='magenta',alpha=0.4)
z_plot.axvline(z_describe[5],color='magenta',alpha=0.4)
z_plot.axvline(z_describe[6],color='magenta',alpha=0.4)


##################################################
#  MIRO EN COSMOS PARA COMPROBAR PICO EN Z=0.5   #
##################################################

plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels

datos_z_cosmos=np.loadtxt('cosmos_z.txt', usecols = (0,3) ,skiprows=1, ) #Lo cargo como str, hay que converir a float
df_cosmos = pd.DataFrame({'z':datos_z_cosmos[:,1]})
dat=((df_cosmos['z'])[df_cosmos.z>0])[df_cosmos.z<0.7]

z_plot=dat.hist(grid=0,ax=ax4,facecolor='silver',edgecolor='black', alpha=0.7)
z_plot.set_xlabel('$Redshift$')
z_plot.set_ylabel('$N$')
z_plot.set_title('COSMOS')
z_describe=dat.describe() #Hago la estadística para el dataset de Z

z_plot.axvline(z_describe[1],color='lime',alpha=0.7)
z_plot.axvline(z_describe[4],color='magenta',alpha=0.4)
z_plot.axvline(z_describe[5],color='magenta',alpha=0.4)
z_plot.axvline(z_describe[6],color='magenta',alpha=0.4)

plt.savefig('figuras/z_ext')
    
#####################################    
### Caracterización de la muestra ###
#####################################
#%%

plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels


ds_fast = ds_fast[ds_fast.zbest<=0.5] #Limito el rango de galaxias de muestra
fig_1, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3, 2,figsize=(8,12))



#Z#


z_plot = ax6 #Dibujo 
dat=ds_z['zbest']
z_plot=dat.hist(grid=0,ax=ax6,facecolor='silver',edgecolor='black', alpha=0.7,bins=binear(dat,6,1))
z_plot.set_xlabel('$Redshift$')
z_describe=dat.describe() #Hago la estadística para el dataset de Z

z_plot.axvline(z_describe[1],color='lime',alpha=0.7)
z_plot.axvline(z_describe[4],color='magenta',alpha=0.4)
z_plot.axvline(z_describe[5],color='magenta',alpha=0.4)
z_plot.axvline(z_describe[6],color='magenta',alpha=0.4)



#Magnitud Absoluta U#



mu_plot = ax5 #Dibujo 
dat = (ds_fast['Mu'][-20<ds_fast.Mu])[ds_fast.Mu<-10]
mu_plot=dat.hist(grid=0,ax=ax5,facecolor='silver',edgecolor='black', alpha=0.7,bins=binear(dat,10,2))
mu_plot.set_xlabel('$M_U \ [mag]$')
mu_plot.set_ylabel('N')

mu_describe=ds_fast['Mu'].describe() #Hago la estadística para el dataset

mu_plot.axvline(mu_describe[1],color='lime',alpha=0.7)
mu_plot.axvline(mu_describe[4],color='magenta',alpha=0.4)
mu_plot.axvline(mu_describe[5],color='magenta',alpha=0.4)
mu_plot.axvline(mu_describe[6],color='magenta',alpha=0.4)



#Magnitud Absoluta V#



dat = ds_fast['Mv'][-30<ds_fast.Mv]

mv_plot = ax4 #Dibujo
mv_plot=dat.hist(grid=0,ax=ax4,facecolor='silver',edgecolor='black', alpha=0.7,bins=binear(dat,10,2))
mv_plot.set_xlabel('$M_V \ [mag]$')


mv_describe=ds_fast['Mv'].describe() #Hago la estadística para el dataset

mv_plot.axvline(mv_describe[1],color='lime',alpha=0.7)
mv_plot.axvline(mv_describe[4],color='magenta',alpha=0.4)
mv_plot.axvline(mv_describe[5],color='magenta',alpha=0.4)
mv_plot.axvline(mv_describe[6],color='magenta',alpha=0.4)



#Magnitud Absoluta J (en el IR)#



mj_plot = ax3 #Dibujo 
data = ds_fast['Mj'][-30<ds_fast.Mj]
mj_plot=dat.hist(grid=0,ax=ax3,facecolor='silver',edgecolor='black', alpha=0.7,bins=binear(dat,10,2))
mj_plot.set_xlabel('$M_J \ [mag]$')
mj_plot.set_ylabel('N')

mj_describe=ds_fast['Mj'].describe() #Hago la estadística para el dataset
mj_plot.axvline(mj_describe[1],color='lime',alpha=0.7)
mj_plot.axvline(mj_describe[4],color='magenta',alpha=0.4)
mj_plot.axvline(mj_describe[5],color='magenta',alpha=0.4)
mj_plot.axvline(mj_describe[6],color='magenta',alpha=0.4)



#Color (U-V)#



col_plot = ax2 #Dibujo 
data = (ds_fast['Mu'][-30<ds_fast.Mu]-ds_fast['Mv'][-30<ds_fast.Mu])
col_plot=data.hist(grid=0,ax=ax2,facecolor='silver',edgecolor='black', alpha=0.7,bins=binear(data,10,2))
col_plot.set_xlabel('$U-V \ [mag]$')


col_describe=(ds_fast['Mu'][-30<ds_fast.Mu]-ds_fast['Mv'][-30<ds_fast.Mu]).describe() #Hago la estadística para el dataset de Z
col_plot.axvline(col_describe[1],color='lime',alpha=0.7)
col_plot.axvline(col_describe[4],color='magenta',alpha=0.4)
col_plot.axvline(col_describe[5],color='magenta',alpha=0.4)
col_plot.axvline(col_describe[6],color='magenta',alpha=0.4)



#Petrosian Radius (px)#
#Escala de placa para el detector IR del WFC3 es 0.1354x0.1209 ''/px#



datos_gen=np.loadtxt('general_3.txt', usecols = (0,77) ,skiprows=1) #Cargo los datos
ds_gen = pd.DataFrame({'id': datos_gen[:, 0], 'petro_rad': datos_gen[:, 1],'zbest':datos_fast[:, 1]} )  #Creo un dataframe de pandas
                     
ds_gen_sel= ds_gen[ds_gen.zbest <= 0.5] #Selecciono rango correcto de z

tam_plot = ax1 #Dibujo todo
data = ds_gen_sel.petro_rad
data = data[data>0]
tam_plot=data.hist(grid=0,ax=ax1,facecolor='silver',edgecolor='black', alpha=0.7,bins=binear(data,10,2))
tam_plot.set_xlabel('$Petrosian \ Radius \ [px]$')
tam_plot.set_ylabel('N')

tam_describe=data.describe() #Hago la estadística para el dataset
tam_plot.axvline(tam_describe[1],color='lime',alpha=0.7)
tam_plot.axvline(tam_describe[4],color='magenta',alpha=0.4)
tam_plot.axvline(tam_describe[5],color='magenta',alpha=0.4)
tam_plot.axvline(tam_describe[6],color='magenta',alpha=0.4)

####
fig_1.savefig('figuras/caracter_1', bbox_inches='tight')
fig_2, ((ax7,ax8),(ax9,ax10),(ax11,ax12)) = plt.subplots(3, 2,figsize=(8,12))
####

#Para Masa



data=(ds_fast[ds_fast.zbest<=0.5]).lmass
mass_plot = ax7 #Dibujo todo
mass_plot=data.hist(grid=0,ax=ax7,facecolor='silver',edgecolor='black', alpha=0.7,bins=binear(data,10,2))
mass_plot.set_xlabel('$log(M/M_{sun})$')
mass_plot.set_ylabel('N')

mass_describe=data.describe() #Hago la estadística para el dataset
mass_plot.axvline(mass_describe[1],color='lime',alpha=0.7)
mass_plot.axvline(mass_describe[4],color='magenta',alpha=0.4)
mass_plot.axvline(mass_describe[5],color='magenta',alpha=0.4)
mass_plot.axvline(mass_describe[6],color='magenta',alpha=0.4)



#Para SFR



data=(ds_fast[ds_fast.zbest<=0.5]).lsfr
sfr_plot = ax8 #Dibujo todo
sfr_plot=data.hist(grid=0,ax=ax8,facecolor='silver',edgecolor='black', alpha=0.7,bins=binear(data,10,2))
sfr_plot.set_xlabel('$log(SFR(M_{sun}/yr))$')

sfr_describe=data.describe() #Hago la estadística para el dataset
sfr_plot.axvline(sfr_describe[1],color='lime',alpha=0.7)
sfr_plot.axvline(sfr_describe[4],color='magenta',alpha=0.4)
sfr_plot.axvline(sfr_describe[5],color='magenta',alpha=0.4)
sfr_plot.axvline(sfr_describe[6],color='magenta',alpha=0.4)



#Para Extinción



datos_ext_tot=np.loadtxt('pobestelar_fast.txt', usecols = (1,5) ,skiprows=1, )
datos_ext_tot = pd.DataFrame({'Av': datos_ext_tot[:, 1],'z': datos_ext_tot[:, 0]})
data = datos_ext_tot.Av[datos_ext_tot.z<=0.5]
ext_plot = ax9 #Dibujo todo
ext_plot=data.hist(grid=0,ax=ax9,facecolor='silver',edgecolor='black', alpha=0.7,bins=binear(data,10,2))
ext_plot.set_xlabel('Av [mag]')
ext_plot.set_ylabel('N')


ext_describe=data.describe() #Hago la estadística para el dataset
ext_plot.axvline(ext_describe[1],color='lime',alpha=0.7)
ext_plot.axvline(ext_describe[4],color='magenta',alpha=0.4)
ext_plot.axvline(ext_describe[5],color='magenta',alpha=0.4)
ext_plot.axvline(ext_describe[6],color='magenta',alpha=0.4)



#Para tau



tau_plot = ax10 #Dibujo  
bin_int = binear(ds_fast['ltau'][ds_fast.zbest<=0.5],7,3)
tau_plot=ds_fast['ltau'][ds_fast.zbest<=0.5].hist(grid=0,ax=ax10,facecolor='silver',edgecolor='black', alpha=0.7,bins=bin_int)
tau_plot.set_xlabel('$log(tau/yr)$')
tau_plot.set_xticks(bin_int)
tau_describe=ds_fast['ltau'][ds_fast.zbest<=0.5].describe() #Hago la estadística para el dataset de Z

tau_plot.axvline(tau_describe[1],color='lime',alpha=0.7)
tau_plot.axvline(tau_describe[4],color='magenta',alpha=0.4)
tau_plot.axvline(tau_describe[5],color='magenta',alpha=0.4)
tau_plot.axvline(tau_describe[6],color='magenta',alpha=0.4)



#Para Edad



age_plot = ax11 #Dibujo 
bin_int = binear(ds_fast['lage'][ds_fast.zbest<=0.5],7,2)
age_plot=ds_fast['lage'][ds_fast.zbest<=0.5].hist(grid=0,ax=ax11,facecolor='silver',edgecolor='black', alpha=0.7,bins=bin_int)
age_plot.set_xlabel('$log(age/yr)$')
age_plot.set_xticks(bin_int)
age_plot.set_ylabel('N')

age_describe=ds_fast['lage'][ds_fast.zbest<=0.5].describe() #Hago la estadística para el dataset de Z

age_plot.axvline(age_describe[1],color='lime',alpha=0.7)
age_plot.axvline(age_describe[4],color='magenta',alpha=0.4)
age_plot.axvline(age_describe[5],color='magenta',alpha=0.4)
age_plot.axvline(age_describe[6],color='magenta',alpha=0.4)



#Z#



z_plot = ax12 #Dibujo 
z_plot=ds_fast['zbest'][ds_fast.zbest<=0.5].hist(grid=0,ax=ax12,facecolor='silver',edgecolor='black', alpha=0.7,bins=binear(ds_fast['zbest'][ds_fast.zbest<=0.5],6,1))
z_plot.set_xlabel('$Redshift$')

z_describe=ds_z['zbest'].describe() #Hago la estadística para el dataset de Z

z_plot.axvline(z_describe[1],color='lime',alpha=0.7)
z_plot.axvline(z_describe[4],color='magenta',alpha=0.4)
z_plot.axvline(z_describe[5],color='magenta',alpha=0.4)
z_plot.axvline(z_describe[6],color='magenta',alpha=0.4)


fig_2.savefig('figuras/caracter_2', bbox_inches='tight')

#%%
##################################
##CORRELACIONES ENTRE PARÁMETROS##
##################################


ds_correl = pd.DataFrame({'id': datos_fast[:, 0], 'zbest': datos_fast[:, 1],'lmass': datos_fast[:, 5], 'lsfr': datos_fast[:, 6],
                          'Mu': datos_fast[:, 8],'Mv': datos_fast[:, 9],'Mj': datos_fast[:, 10],
                     'ltau': datos_fast[:, 2],'metal': datos_fast[:, 3],'lage': datos_fast[:, 4],
                     'la2t': datos_fast[:, 7],
                     'petro_rad': datos_gen[:, 1],
                     'Av': datos_ext_tot['Av']})

ds_obs = pd.DataFrame({'id': datos_fast[:, 0],'zbest': datos_fast[:, 1],'Mu': datos_fast[:, 8],'Mv': datos_fast[:, 9],'Mj': datos_fast[:, 10],'petro_rad': datos_gen[:, 1],
                     'Av': datos_ext_tot['Av']})#Parámetros observacionales
ds_deriv = pd.DataFrame({'id': datos_fast[:, 0], 'zbest': datos_fast[:, 1], 
                         'lmass': datos_fast[:, 5],'lsfr': datos_fast[:, 6],
                     'ltau': datos_fast[:, 2],'metal': datos_fast[:, 3],'lage': datos_fast[:, 4],
                     'la2t': datos_fast[:, 7]})#Parámetros derivados

ds_correl = ds_correl[ds_correl.zbest<=0.5] #Ajusto al rango de z válidos
ds_obs = ds_obs[ds_obs.zbest<=0.5]
ds_deriv = ds_deriv[ds_deriv.zbest<=0.5]

ds_correl = ds_correl[ds_correl.lmass>6] #Ajusto al rango de lmass válidos
ds_obs = ds_obs[ds_deriv.lmass>6]
ds_deriv = ds_deriv[ds_deriv.lmass>6]

ds_correl = ds_correl[-40<ds_correl.Mv] #Ajusto al rango de Mv válidos
ds_deriv = ds_deriv[-40<ds_obs.Mv]
ds_obs = ds_obs[-40<ds_obs.Mv]

ds_correl = ds_correl[-7<ds_correl.lsfr] #Ajusto al rango de lsfr válidos (cambia mucho correlaciones)
ds_obs = ds_obs[-7<ds_deriv.lsfr]
ds_deriv = ds_deriv[-7<ds_deriv.lsfr]


corr_gen = np.abs(ds_correl.corr())
corr_obs = np.abs(ds_obs.corr()) #Correlación con todos los parámetros del dataframe
corr_deriv = np.abs(ds_deriv.corr())

#### Mapa de Correlaciones ####


plt.matshow(corr_gen,cmap='coolwarm')
plt.xticks(range(ds_correl.select_dtypes(['number']).shape[1]), ds_correl.select_dtypes(['number']).columns, fontsize=12, rotation=90)
plt.yticks(range(ds_correl.select_dtypes(['number']).shape[1]), ds_correl.select_dtypes(['number']).columns, fontsize=12, rotation=0)
cb = plt.colorbar()
plt.figtext(0.5, 0.03, '', fontsize = 12,horizontalalignment ="center")
plt.savefig('figuras/correl_gen')

plt.matshow(corr_obs,cmap='coolwarm')
plt.xticks(range(ds_obs.select_dtypes(['number']).shape[1]), ds_obs.select_dtypes(['number']).columns, fontsize=12, rotation=90)
plt.yticks(range(ds_obs.select_dtypes(['number']).shape[1]), ds_obs.select_dtypes(['number']).columns, fontsize=12, rotation=0)
cb = plt.colorbar()
plt.figtext(0.5, 0.03, 'Parámetros Observacionales', fontsize = 12,horizontalalignment ="center")
plt.savefig('figuras/correl_obs')

plt.matshow(corr_deriv,cmap='coolwarm')
plt.xticks(range(ds_deriv.select_dtypes(['number']).shape[1]), ds_deriv.select_dtypes(['number']).columns, fontsize=12, rotation=90)
plt.yticks(range(ds_deriv.select_dtypes(['number']).shape[1]), ds_deriv.select_dtypes(['number']).columns, fontsize=12, rotation=0)
cb = plt.colorbar()
plt.figtext(0.5, 0.03, 'Parámetros Derivados', fontsize = 12,horizontalalignment ="center")
plt.savefig('figuras/correl_deriv')

#%%

####MOSAICO DE GRÁFICOS####

########
def plot_correl(x,y,xlabel,ylabel,ax): #Función para dibujar correlaciones
    ax.plot(x,y,'.',color='grey')
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    slope, intercept, r, p, stderr = stats.linregress(x, y)
    line = f'y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
    ax.plot(x, intercept + slope * x,'k' ,label=line,alpha=.3)  
    ax.legend(facecolor='silver')
    ax.legend(loc=2, prop={'size': 16})
    plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
    
def plot_correl_noax(x,y,xlabel,ylabel): #Función para dibujar sin eje predispuesto
    plt.figure()
    ax = plt.subplot()
    ax.plot(x,y,'.',color='grey')
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    slope, intercept, r, p, stderr = stats.linregress(x, y)
    line = f'y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
    ax.plot(x, intercept + slope * x,'k' ,label=line,alpha=.3)  
    ax.legend(facecolor='silver')
    ax.legend(loc=2, prop={'size': 16})
    plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
    
def plot_correl_scat(x,y,col,xlabel,ylabel,zlabel,size=None):
    sc=plt.scatter(x, y,c=col,s=size, alpha=0.3,cmap='plasma')
    cbar=plt.colorbar(sc)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    cbar.set_label(zlabel, fontsize=18)
    plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
########
fig_3, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2,figsize=(18,12))
fig_4,((ax5,ax6),(ax7,ax8)) =plt.subplots(2, 2,figsize=(18,12))

plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
#### lmass vs z ####
x = ds_correl.lmass
y = ds_correl.zbest
z_lmass = plot_correl(x,y,'$log(M/M_{sun})$','$redshift$',ax1)
#plt.savefig('figuras/z_vs_M')
##### lmass vs sfr #####
x = ds_correl.lmass
y = ds_correl.lsfr
z_lmass = plot_correl(x,y,'$log(M/M_{sun})$','$log [ SFR \cdot (M_{sun}/yr)^{-1} ]$',ax2)
plt.ylim(-4,6)
#plt.savefig('figuras/z_vs_sfr')
##### lmass vs rad #####
x = ds_correl.lmass[ds_correl.petro_rad>0.2]
y = ds_correl.petro_rad[ds_correl.petro_rad>0.2]
z_lmass = plot_correl(x,y,'$log(M/M_{sun})$','petrosian radius [px]',ax3)
#plt.savefig('figuras/z_vs_rad')
##### lmass vs tau #####
x = ds_correl.lmass
y = ds_correl.ltau
z_lmass = plot_correl(x,y,'$log(M/M_{sun})$','$log(tau / yr)$',ax4)
fig_3.savefig('figuras/lmass_vs_all_1',bbox_inches='tight')
#plt.savefig('figuras/z_vs_tau')
##### lmass vs color #####
x = ((ds_correl.lmass)[-40<ds_correl.Mu])[-40<ds_correl.Mv]
y = ((ds_correl['Mu']-ds_correl['Mv'])[-40<ds_correl.Mu])[-40<ds_correl.Mv]
z_lmass = plot_correl(x,y,'$log(M/M_{sun})$','$U-V \ [mag]$',ax5)
#plt.savefig('figuras/z_vs_color')
##### lmass vs Mv #####
x = ((ds_correl.lmass))[ds_correl.Mv > -80]
y = ((ds_correl['Mv']))[ds_correl.Mv > -80]
z_lmass = plot_correl(x,y,'$log(M/M_{sun})$','$M_V \ [mag]$',ax6)
#plt.savefig('figuras/z_vs_A_V')
##### lmass vs Av #####
x = ((ds_correl.lmass))
y = ((ds_correl['Av']))
z_lmass = plot_correl(x,y,'$log(M/M_{sun})$','$A_V \ [mag]$',ax8)
##### lmass vs edad #####
x = ((ds_correl.lmass))
y = ((ds_correl['lage']))
z_lmass = plot_correl(x,y,'$log(M/M_{sun})$','$log(age/yr)$',ax7)

fig_4.savefig('figuras/lmass_vs_all_2',bbox_inches='tight')


#%% #Para sacar r de correlación lsfr vs zbest
x = ((ds_correl.lsfr))
y = ((ds_correl['zbest']))
slope, intercept, r, p, stderr = stats.linregress(x, y)
#%% #Correlación Mv vs SFR
x = ((ds_correl.lsfr))
y = ((ds_correl['Mv']))
print(stats.linregress(x, y))
#%% #Correlación Radio vs Mv
x = ((ds_correl.petro_rad))
y = ((ds_correl['Mv']))
print(stats.linregress(x, y))
#%% #Correlación z vs SFR
x = ((ds_correl.zbest))
y = ((ds_correl['lsfr']))
print(stats.linregress(x, y))
#%%
#####GRÁFICOS EN COLOR######
#fig_5, axs = plt.subplots(1, 2,figsize=(20,8))
##### sfr vs Mv vs rad #####
#plt.sca(axs[0])
plt.figure()
x = ds_correl.lsfr[ds_correl.Mv > -80]
y = ds_correl.lmass[ds_correl.Mv > -80]
z = ds_correl.zbest[ds_correl.Mv > -80]
plot_correl_scat(x,y,z,'log(SFR)','$M_V \ [mag]$','petrosian radius [px]',size=z*7)
#plt.savefig('figuras/Z_vs_Mv_rad_corr_3d')
plt.savefig('figuras/sfr_mv_rad', bbox_inches='tight')

##### z frente vs sfr vs Mv #####
#plt.sca(axs[1])
plt.figure()
x = ds_correl.lsfr[ds_correl.Mv > -80]
y = ds_correl.Mv[ds_correl.Mv > -80]
z = ds_correl.zbest[ds_correl.Mv > -80]
plot_correl_scat(x,y,z,'log(SFR)','$M_V \ [mag]$','redshift') 

plt.savefig('figuras/sfr_mv_z', bbox_inches='tight')

##### z frente vs petro_rad vs Mv #####
#plt.sca(axs[1])
plt.figure()
x = ds_correl.petro_rad[ds_correl.Mv > -80][ds_correl.petro_rad>1]
y = ds_correl.Mv[ds_correl.Mv > -80][ds_correl.petro_rad>1]
z = ds_correl.zbest[ds_correl.Mv > -80][ds_correl.petro_rad>1]
plot_correl_scat(x,y,z,'petrosian radius [px]','$M_V \ [mag]$','redshift') 

plt.savefig('figuras/rad_mv_z', bbox_inches='tight')
#%%
#### color vs Mv vs masa ####
x = ((ds_correl['Mu']-ds_correl['Mj'])[-40<ds_correl.Mu])[-40<ds_correl.Mv]
y = ((ds_correl.Mv)[-80<ds_fast.Mu])[-40<ds_correl.Mv]
z = ((ds_correl.lmass)[-80<ds_fast.Mu])[-40<ds_correl.Mv]

data = pd.concat([x,z,y],axis=1,keys=['color','mass','mag'])

y = ((y)[x<3])[x>-1]
z = ((z)[x<3])[x>-1]
x = ((x)[x<3])[x>-1]

# plt.figure(figsize=(10,8))
# dibu=sns.kdeplot(data=data,x='mass',y='color',fill=True,levels=20,tresh=0.1,cmap='flare')
# dibu.set(ylim=[-1,3],xlim=[5,12])
# dibu.set_xlabel('$log(M/M_{sun})$', fontsize = 18)
# dibu.set_ylabel('$U-V \ [mag]$', fontsize = 18)
# plt.savefig('figuras/col_mag', bbox_inches='tight')

plt.figure(figsize=(10,8))
dibu=sns.kdeplot(data=data,y='color',x='mag',fill=False,levels=15)
dibu.set(ylim=[-3,6],xlim=[-8,-26])
dibu.set_xlabel('$M_V \ [mag]$', fontsize = 18)
dibu.set_ylabel('$U-J \ [mag]$', fontsize = 18)
plt.savefig('figuras/col_mag', bbox_inches='tight')



#%%
######### ANÁLISIS NO PARAMÉTRICO ############

v_r = np.array([]) #vector para almacenar valores de r
v_kendall= np.array([]) #para coef. kendall
v_spearman = np.array([]) #para coef. spearman

#lmass - sfr
print('lmass - sfr')
lmass = ds_correl.lmass
lsfr = ds_correl.lsfr


slope, intercept, r, p, stderr = stats.linregress(lmass, lsfr) 
v_r = np.append(v_r,r)

print('kendall')

t,p = stats.kendalltau(lmass,lsfr)
print(t)
print(p>0.05) #si es false se rechaza H0, o lo que es igual, hay correlación

print('spearmanr')

t,p = stats.spearmanr(lmass,lsfr)
print(t)
print(p>0.05) #si es false se rechaza H0, o lo que es igual, hay correlación

ts = TheilSenRegressor(random_state=0).fit(np.array(lmass.values.reshape(-1, 1)), np.array(lsfr.values.reshape(-1, 1)))
print(ts.coef_[0],ts.intercept_)

#lmass - mv
print('lmass - mv')
mv= ds_correl.Mv
lmass_1= ds_correl.lmass

slope, intercept, r, p, stderr = stats.linregress(lmass, mv) 
v_r = np.append(v_r,r)

print('kendall')

t,p = stats.kendalltau(lmass,mv)
print(t)
print(p>0.05) #si es false se rechaza H0, o lo que es igual, hay correlación


print('spearmanr')

t,p = stats.spearmanr(lmass,mv)
print(t)
print(p>0.05) #si es false se rechaza H0, o lo que es igual, hay correlación

ts = TheilSenRegressor(random_state=0).fit(lmass_1.values.reshape(-1, 1), mv.values.reshape(-1, 1))
print(ts.coef_[0],ts.intercept_)

#lmass - lage
print('lmass - lage')
lmass_2= ds_correl.lmass
lage= ds_correl.lage

slope, intercept, r, p, stderr = stats.linregress(lmass, lage) 
v_r = np.append(v_r,r)

print('kendall')

t,p = stats.kendalltau(lmass,lage)
print(t)
print(p>0.05) #si es false se rechaza H0, o lo que es igual, hay correlación

print('spearmanr')

t,p = stats.spearmanr(lmass,lage)
print(t)
print(p>0.05) #si es false se rechaza H0, o lo que es igual, hay correlación

ts = TheilSenRegressor(random_state=0).fit(lmass_2.values.reshape(-1, 1), lage.values.reshape(-1, 1))
print(ts.coef_[0],ts.intercept_)

#z - lsfr
print('z - lsfr')
z= ds_correl.zbest
lsfr= ds_correl.lsfr

slope, intercept, r, p, stderr = stats.linregress(z, lsfr) 
v_r = np.append(v_r,r)

print('kendall')

t,p = stats.kendalltau(z,lsfr)
print(t)
print(p>0.05) #si es false se rechaza H0, o lo que es igual, hay correlación

print('spearmanr')

t,p = stats.spearmanr(z,lsfr)
print(t)
print(p>0.05) #si es false se rechaza H0, o lo que es igual, hay correlación

ts = TheilSenRegressor(random_state=0).fit(z.values.reshape(-1, 1), lsfr.values.reshape(-1, 1))
print(ts.coef_[0],ts.intercept_)



print(v_r)


#%%
###### Matrices de Correlación  #######


corr_kendall = abs(ds_correl.corr(method='kendall'))

corr_spearman = abs(ds_correl.corr(method='spearman'))






plt.matshow(corr_kendall,cmap='coolwarm')
plt.xticks(range(ds_correl.select_dtypes(['number']).shape[1]), ds_correl.select_dtypes(['number']).columns, fontsize=12, rotation=90)
plt.yticks(range(ds_correl.select_dtypes(['number']).shape[1]), ds_correl.select_dtypes(['number']).columns, fontsize=12, rotation=0)
plt.title('$tau$ de Kendall')
cb = plt.colorbar()
plt.figtext(0.5, 0.03, '', fontsize = 14,horizontalalignment ="center")

plt.savefig('figuras/kendall', bbox_inches='tight')



plt.matshow(corr_spearman,cmap='coolwarm')
plt.xticks(range(ds_correl.select_dtypes(['number']).shape[1]), ds_correl.select_dtypes(['number']).columns, fontsize=12, rotation=90)
plt.yticks(range(ds_correl.select_dtypes(['number']).shape[1]), ds_correl.select_dtypes(['number']).columns, fontsize=12, rotation=0)
plt.title('$r$ de Spearman')
cb = plt.colorbar()
plt.figtext(0.5, 0.03, '', fontsize = 14,horizontalalignment ="center")

plt.savefig('figuras/spear', bbox_inches='tight')

#%%
####### tamaño físico ########
#Tenemos que escala de placa es 0.13"\px para banda F160W con cámara WFC3

d_lum = Distance(z=ds_correl.zbest,cosmology=WMAP9) #distancia de luminosidad en MPc


#Ahora la convierto a cordenada r, y hago que theta=radio objeto/distancia 

d = d_lum/(1+ds_correl.zbest)

#Finalemnte se tiene que:

theta = ds_correl.petro_rad*0.13/3600*np.pi/180 #ángulo en rad

tam = d*theta*1000 #tamaño en KPc

plt.plot(tam,d_lum,'.',alpha=0.5,color='grey')
plt.xlabel('tamaño [kpc]', fontsize=20)
plt.ylabel('distancia luminosidad [Mpc]', fontsize=20)
plt.savefig('figuras/tam_dist', bbox_inches='tight')

#%% Dibujo correlaciones sencillas
x = tam
y = ds_correl.Mv
z_lmass = plot_correl_noax(x,y,'$petrosian \ radius \ [kpc]$','$M_V \ [mag]$')
plt.savefig('figuras/petro_mv', bbox_inches='tight')

x = tam
y = ds_correl.zbest
z_lmass = plot_correl_noax(x,y,'$petrosian \ radius \ [kpc]$','$z$')
plt.savefig('figuras/petro_z', bbox_inches='tight')

x = tam
y = ds_correl.lmass
z_lmass = plot_correl_noax(x,y,'$petrosian \ radius \ [kpc]$','$log(M/M_{sun})$')
plt.savefig('figuras/petro_lmass', bbox_inches='tight')

#%%  Versiones en 3d

plt.figure()
x = tam
y = ds_correl.zbest
z = ds_correl.lsfr
plot_correl_scat(x,z,y,'tamaño [kpc]','$log(SFR/(M_{sun}/yr))$','$z$') 

plt.savefig('figuras/lsfr_z_petro', bbox_inches='tight')


plt.figure()
x = tam
y = ds_correl.Mv
z = ds_correl.lmass
plot_correl_scat(x,y,z,'tamaño [kpc]','$M_V \ [mag]$','$log(M/M_{sun})$') 

plt.savefig('figuras/lmass_mv_petro', bbox_inches='tight')





#%%
########### COMPARACIÓN ENTRE COMPAÑEROS ##############

fig_5,(ax1,ax2,ax3) = plt.subplots(1, 3,figsize=(20,5))

y = ds_correl.lsfr
x = ds_correl.lmass
plot_correl(x,y,'$log(M/M_{sun})$','$log(SFR)$',ax=ax1)
ax1.set_title('z=0 | z=0.5')

plt.savefig('figuras/lsfr_lmass', bbox_inches='tight')




datos_comp_1=np.loadtxt('pobestelar_fast.txt', usecols = (0,1,2,3,4,6,7,9,11,12,13) ,skiprows=1, ) #Lo cargo como str, hay que converir a float
selec_comp_1=np.zeros((cuenta_rango(datos_comp_1[:,1].tolist(), 1, 1.5),11)) #Creo objeto para alacenar datos filtrados
extrae_rango(datos_comp_1,1,selec_comp_1,1,1.5)

ds_comp_1 = pd.DataFrame({'id': selec_comp_1[:, 0], 'zbest': selec_comp_1[:, 1],    #datos de galaxias hasta z=0.5
                     'ltau': selec_comp_1[:, 2],'metal': selec_comp_1[:, 3],'lage': selec_comp_1[:, 4],
                     'lmass': selec_comp_1[:, 5],'lsfr': selec_comp_1[:, 6],'la2t': selec_comp_1[:, 7],
                     'Mu': selec_comp_1[:, 8],'Mv': selec_comp_1[:, 9],'Mj': selec_comp_1[:, 10]})
y = ds_comp_1.lsfr
x = ds_comp_1.lmass
plot_correl(x,y,'$log(M/M_{sun})$','$log(SFR)$',ax=ax2)
ax2.set_title('z=1 | z=1.5')




datos_comp_1=np.loadtxt('pobestelar_fast.txt', usecols = (0,1,2,3,4,6,7,9,11,12,13) ,skiprows=1, ) #Lo cargo como str, hay que converir a float
selec_comp_1=np.zeros((cuenta_rango(datos_comp_1[:,1].tolist(), 1.5, 3),11)) #Creo objeto para alacenar datos filtrados
extrae_rango(datos_comp_1,1,selec_comp_1,1.5,3)

ds_comp_1 = pd.DataFrame({'id': selec_comp_1[:, 0], 'zbest': selec_comp_1[:, 1],    #datos de galaxias hasta z=0.5
                     'ltau': selec_comp_1[:, 2],'metal': selec_comp_1[:, 3],'lage': selec_comp_1[:, 4],
                     'lmass': selec_comp_1[:, 5],'lsfr': selec_comp_1[:, 6],'la2t': selec_comp_1[:, 7],
                     'Mu': selec_comp_1[:, 8],'Mv': selec_comp_1[:, 9],'Mj': selec_comp_1[:, 10]})
y = ds_comp_1.lsfr
x = ds_comp_1.lmass
plot_correl(x,y,'$log(M/M_{sun})$','$log(SFR)$',ax=ax3)
ax3.set_title('z=1.5 | z=3')


plt.savefig('figuras/compar', bbox_inches='tight')



