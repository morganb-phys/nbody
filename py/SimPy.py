import numpy as np
import matplotlib.pyplot as plt
import struct

from galpy.util.conversion import time_in_Gyr, mass_in_msol
import tqdm
import pynbody as pb
import galpy
from galpy.potential import SnapshotRZPotential
from galpy.potential import SCFPotential, scf_compute_coeffs_nbody
from galpy.util import plot
from galpy.util import conversion
plot.start_print(axes_labelsize=17.,text_fontsize=16.,xtick_labelsize=14.,ytick_labelsize=14.)

class Snapshot():

    MLU= 8. #kpc
    MVU= 220. #km/s
    MTU= time_in_Gyr(MVU,MLU)*1000. #Myr
    MMU= mass_in_msol(MVU,MLU) # msol
    
    def __init__(self,Filename,Np,hf=0.4,df=0.5,bf=0.1,Nslice=10,dt=0.0078125,frac=True):
        
        self.Filename= Filename
        self.Np= Np
        if frac:
            self.Nh= int(self.Np*hf)
            self.Nd= int(self.Np*df)
            self.Nb= int(self.Np*bf)
        else:
            self.Nh= hf
            self.Nd= df
            self.Nb= bf
        self._step= int((self.Filename.split('_')[3]).split('.')[0])
        self._Nslice= Nslice
        self._dt= dt
        self.t= self._step*dt
        
        self._COM= self.calc_COM()
        self._rot= self.calc_AngMom()        
        
    def calc_COM(self):
        mtot= 0
        COMraw= np.zeros(3)
        for i in range(self._Nslice):
            offset=int(i*int(np.ceil(self.Np/self._Nslice)))
            count=int(np.ceil(self.Np/self._Nslice))
            if offset+count>self.Np:
                count= self.Np-offset
            with open(self.Filename,'rb') as f:
                mxv= np.fromfile(f,dtype='f4',offset=offset*4*7,count=count*7)
                mxv= np.reshape(mxv,(count,7)).astype('f8')
                mtot+=np.sum(mxv[:,0])
                COMraw+=np.sum(mxv[:,0][:,None]*(mxv[:,1:4]),axis=0)    
        del mxv
        return COMraw/mtot
        
    def calc_AngMom(self):
        Ltot= np.zeros(3)
        for i in range(self._Nslice):
            with open(self.Filename,'rb') as f:
                mxv= np.fromfile(f,dtype='f4',offset=(self.Nh*7+i*7*int(self.Nd/self._Nslice))*4,
                                 count=7*int(self.Nd/self._Nslice))
                mxv= np.reshape(mxv,(int(len(mxv)/7),7))
                Ltot+= np.sum(np.cross((mxv[:,1:4]-self._COM),mxv[:,0][:,None]*mxv[:,4:]),axis=0)
        Ltot= Ltot/np.sqrt(np.sum(Ltot**2))
        rot= self.calc_Rot_matrix(Ltot) 
        del mxv
        return rot
    
    @staticmethod
    def calc_Rot_matrix(AngMom):
        hyz= np.sqrt(AngMom[1]**2+AngMom[2]**2)
        costh= AngMom[2]/hyz
        sinth= AngMom[1]/hyz

        Rx= np.array([[1,0,0],[0,costh,-sinth],[0,sinth,costh]])

        Lmid= np.matmul(Rx,AngMom)

        hxz= np.sqrt(Lmid[0]**2+Lmid[2]**2)
        cosph= Lmid[2]/hxz
        sinph= -Lmid[0]/hxz
        Ry= np.array([[cosph,0,sinph],[0,1,0],[-sinph,0,cosph]])

        return np.matmul(Ry,Rx)
    
    def calc_density(self,bins=(90,90,90),lim=(30.,30.,10.),adjust=True):
        Ntot= np.zeros(bins)

        for i in range(self._Nslice):
            with open(self.Filename,'rb') as f:
                mxv= np.fromfile(f,dtype='f4',offset=i*7*int(self.Np/self._Nslice)*4,count=7*int(self.Np/self._Nslice))
                mxv= np.reshape(mxv,(int(len(mxv)/7),7))
                if adjust:
                    mxv[:,1:]= self.adjust(mxv[:,1:])

                N, edges= np.histogramdd(mxv[:,1:4], bins=bins, 
                                         range=((-lim[0],lim[0]),(-lim[1],lim[1]),(-lim[2],lim[2])),
                                         weights=mxv[:,0])
                Ntot+=N
        del mxv
        self.N= Ntot
        self.Nedge= edges
        return Ntot, edges
    
    def adjust(self,particles,vel=True):
        particles[:,:3]= np.matmul(self._rot,particles[:,:3].T-self._COM[:,None]).T
        if vel:
            particles[:,3:]= np.matmul(self._rot,particles[:,3:].T).T
        return particles
    
    def sample(self,frac=0.1):
        
        ncomp= [self.Nh,self.Nd,self.Nb]
        sample= np.empty([0,7])
        
        for i,n in enumerate(ncomp):    
            for j in range(self._Nslice):
                offset=int(np.sum(ncomp[:i]))+j*int(np.ceil(n/self._Nslice))
                count=int(np.ceil(n/self._Nslice))
                if offset+count>np.sum(ncomp[:i+1]):
                    count= np.sum(ncomp[:i+1])-offset
                with open(self.Filename,'rb') as f:
                    mxv= np.fromfile(f,dtype='f4',offset=offset*7*4,
                                     count=7*count)
                    mxv= np.reshape(mxv,(int(len(mxv)/7),7))    
                    indx= np.random.choice(count,int(count*frac),replace=False)
                    sample=np.vstack([sample,np.reshape(mxv[indx],(len(indx),7))])
        sample[:,0]/=frac
        sample[:,1:]= self.sample(frac=frac)
        return sample
    
    def calc_vrot(self,Rs=np.linspace(0.01,5.,51),sample=True,frac=0.01):
        
        try:
            return self.vrot
        except:
            if sample:
                sample= self.sample(frac=frac)
            else:
                with open(self.Filename,'rb') as f:
                    sample= np.fromfile(f,dtype='f4',offset=4*7*(self.Nh+self.Nd),count=7*self.Nb)
                    sample= np.reshape(sample,(self.Nb,7))
                    sample= self.adjust(sample) 

            Nh= len(sample[tuple([sample[:,0]==sample[0,0]])])

            f= pb.new(dm=Nh,star=len(sample[:,0])-Nh)
            f['mass']= sample[:,0]/self.MMU
            f['pos']= sample[:,1:4]/self.MLU
            f['vel']= sample[:,4:]/self.MVU
            f['eps']= np.ones(len(sample))*0.05/self.MLU

            sp= galpy.potential.SnapshotRZPotential(f,num_threads=10)
            vrot= galpy.potential.calcRotcurve(sp,Rs,phi=0.)
            self.vrot= vrot
            self._Rs= Rs
            return vrot


    def calc_potential(self,sample=True,frac=0.01,a=[27.68,3.41,0.536],N= [5,15,3],L= [5,15,3]):
        
        try:
            print('data/equil_potential_coefficients_N'+str(N[0])+str(N[1])+str(N[2])+'.txt')
            with open('data/equil_potential_coefficients_N'+str(N[0])+str(N[1])+str(N[2])+'.txt','rb') as f:
                Acos,Asin= np.load(f,allow_pickle=True)
            pot= [SCFPotential(Acos=Acos[i],Asin=Asin[i],a=a[i]/8.) for i in range(3)]
        except:   
            ncomp= [self.Nh,self.Nd,self.Nb]
            Acos,Asin= np.empty((2,3),dtype='object')

            pot= np.empty(3,dtype='object')

            fullsample= self.sample(frac=frac)
            masses= list(set(fullsample[:,0]))

            for i,n in enumerate(ncomp):
                samples= fullsample[tuple([fullsample[:,0]==masses[i]])]
                Acos[i], Asin[i]= scf_compute_coeffs_nbody(samples[:,1:4].T/self.MLU,
                                                           samples[:,0]/self.MMU,N[i],L[i],a=a[i]/8.)
                pot[i]= SCFPotential(Acos=Acos[i],Asin=Asin[i],a=a[i]/8.)
            
            coeff= np.vstack([Acos,Asin])
            with open('data/equil_potential_coefficients_N'+str(N[0])+str(N[1])+str(N[2])+'.txt','wb') as f:
                np.save(f,coeff)
                    
        self.pot= list(pot)
        return list(pot)
    
    def analyze_SN(self,Rcyl,X=8.1,Y=0,correct=True,zrange=[-2,2],nbins=51):
       
        calc=True
        try:
            for i,sn in enumerate(self.SN):
                if (sn.Rcyl==Rcyl)*(sn.X==X)*(sn.Y==Y):
                    
                    # If you want a finer grid, replace the old coarser grid
                    if (nbins>sn.nbins):
                        self.SN[i]= SolarNeighbourhood(self,Rcyl,X,Y,correct,zrange,nbins)
                        calc=False
                    
                    # If it already exists, don't do it again
                    else:
                        calc= False
                else:
                    continue
                    
            if calc:
                self.SN= np.append(self.SN,SolarNeighbourhood(self,Rcyl,X,Y,correct,zrange,nbins))
                        
        except:
            self.SN= np.array([SolarNeighbourhood(self,Rcyl,X,Y,correct,zrange,nbins)])
    
    def plot_density(self,plot_origin=False,save=False,adjust=True):
        
        try:
            self.N
        except:
            self.calc_density(adjust=adjust)
            
        fig,[ax1,ax2]= plt.subplots(2,sharex=True,gridspec_kw={'height_ratios': [3, 1]},figsize=[5,7])
        ax1.imshow(np.log(np.sum(self.N,axis=2).T),cmap='Greys',extent=[-30,30,-30,30])
        ax2.imshow(np.log(np.sum(self.N,axis=1).T),cmap='Greys',extent=[-30,30,-10,10])
        
        if plot_origin:
            ax1.plot(0.,0.,'or')
            ax2.plot(0.,0.,'or')
        
        ax2.set_xlabel(r'$x \,\mathrm{(kpc)}$')
        ax1.set_ylabel(r'$y \,\mathrm{(kpc)}$')
        ax2.set_ylabel(r'$z \,\mathrm{(kpc)}$')
        fig.subplots_adjust(hspace=0)
        txt= ax1.annotate(r'$t=%.0f\,\mathrm{Myr}$' % (np.round(self.t,0)),
                     (0.95,0.95),xycoords='axes fraction',
                     horizontalalignment='right',verticalalignment='top',size=18.)
        if save:
            plt.savefig('plots/Density'+str(np.round(self.t,0)+'.pdf',bbox_inches='tight'))
    
    
class SolarNeighbourhood():
    
    def __init__(self,Snapshot,Rcyl,X=8.1,Y=0,correct=True,zrange=[-2,2],nbins=51):
        
        self.Snapshot= Snapshot # Simulation snapshot to be analyzed
        self.Rcyl= Rcyl # Radius of solar neighbourhood cylinder in kpc
        self.X= X
        self.Y= Y
        self.phi= np.arctan2(Y,X)
        self.nbins=nbins
        self.zrange= zrange
        
        self.particles= self.calc_particles() # mass, position, and velocity of solar neighbourhood particles
        
        self.z,self.n,self.n_err= self.calc_numberdensity() # Heights and number density
        self.N= self.calc_numbercount()      # Particle Number count 
        self.zA,self.A,self.A_err= self.calc_A()      # Asymmetry Heights, Asymmetry, and Asymmetry uncertainty
    
    @property
    def sliceVolume(self):
        return (self.zrange[1]-self.zrange[0])*1000./self.nbins*np.pi*(self.Rcyl*1000.)**2
        
    def calc_particles(self):
        
        particles=np.empty((0,7))
        for i in range(self.Snapshot._Nslice):
            with open(self.Snapshot.Filename,'rb') as f:
                mxv= np.fromfile(f,dtype='f4',
                                 offset=(self.Snapshot.Nh+i*int(self.Snapshot.Nd/self.Snapshot._Nslice))*4*7,
                                 count=7*int(self.Snapshot.Nd/self.Snapshot._Nslice))
                mxv= np.reshape(mxv,(int(len(mxv)/7),7))

                mask= tuple([(mxv[:,1]-self.X)**2+(mxv[:,2]-self.Y)**2<self.Rcyl**2])

                newparticles= mxv[mask]
                particles= np.vstack([particles,newparticles])
                
        del mxv
        return particles
    
    def calc_numberdensity(self):
        n, edge= np.histogram(self.particles[:,3]-np.median(self.particles[:,3]),
                              bins=self.nbins,range=self.zrange,weights=self.particles[:,0])
        n_err, edge= np.histogram(self.particles[:,3]-np.median(self.particles[:,3]),
                                  bins=self.nbins,range=self.zrange,weights=self.particles[:,0]**2)
        mid= edge[:-1]+np.diff(edge)/2
        return mid,n/self.sliceVolume,np.sqrt(n_err)/self.sliceVolume
    
    def calc_numbercount(self):
        return np.histogram(self.particles[:,3]-np.median(self.particles[:,3]),
                            bins=self.nbins,range=self.zrange)[0]  
    
    def calc_A(self):
        A= (self.n-self.n[::-1])/(self.n+self.n[::-1])
        A_err= np.sqrt((2*self.n/(self.n+self.n[::-1])**2*self.n_err)**2+
                       (2*self.n[::-1]/(self.n+self.n[::-1])**2*self.n_err[::-1])**2)
        return self.z[self.z>=0],A[self.z>=0],A_err[self.z>=0]
    

def analyze_ring(mass,position,velocity,rin,rout,correct=True,nd=0,zrange=[-2,2]):
    # Calculate angular momentum of the ring
    if nd==0:
        nd= int(0.6*len(mass))
    if np.shape(position)[0]==3:
        pass
    else:
        position= position.T
        velocity= velocity.T    
    
    mask= np.reshape([(position[0]**2+position[1]**2>rin**2)*(position[0]**2+position[1]**2<rout**2)],nd)
    xx= position[:,mask]
    vv= velocity[:,mask]
    mm= np.reshape(mass[mask],len(xx.T))
    
    if correct:
        xx,vv= AlignDisc(mm,xx.T,vv.T,len(mass))

    nbins= 51
    volume= (zrange[1]-zrange[0])*1000./nbins*np.pi*((1000.*rout)**2-(1000*rin)**2)
    z,bins= np.histogram(xx[2]-np.median(xx[2]),bins=nbins,range=zrange,weights=mm)
    z= z/volume
    N,bins= np.histogram(xx[2]-np.median(xx[2]),bins=nbins,range=zrange)
    z_err2,bins= np.histogram(xx[2],bins=nbins,range=zrange,weights=mm**2)
    mid= bins[:-1]+np.diff(bins)/2
    z_err= np.sqrt(z_err2)/volume
    
    A= (z-z[::-1])/(z+z[::-1])
    A_err= np.sqrt((2*z/(z+z[::-1])**2*z_err)**2+(2*z[::-1]/(z+z[::-1])**2*z_err[::-1])**2)

    return mid,z,z_err,N,mid[mid>0],A[mid>0],A_err[mid>0]

def _loader(filename):
    ffile = open(filename, 'rb')
    t,n,ndim,ng,nd,ns,on=struct.unpack("<diiiiii",ffile.read(32))

    catd = {'mass':np.zeros(nd), 'x':np.zeros(nd), 'y':np.zeros(nd),'z':np.zeros(nd),'vx':np.zeros(nd),'vy':np.zeros(nd),'vz':np.zeros(nd),'ID':np.zeros(nd)}
    cats = {'mass':np.zeros(ns), 'x':np.zeros(ns), 'y':np.zeros(ns),'z':np.zeros(ns),'vx':np.zeros(ns),'vy':np.zeros(ns),'vz':np.zeros(ns),'metals':np.zeros(ns), 'tform':np.zeros(ns), 'ID':np.zeros(ns)}

    for i in range(nd):
        mass, x, y, z, vx, vy, vz, IDs = struct.unpack("<fffffffQ", ffile.read(36))
        catd['mass'][i] = mass*2.324876e9
        catd['x'][i] = x
        catd['y'][i] = y
        catd['z'][i] = z
        catd['vx'][i] = vx*100.
        catd['vy'][i] = vy*100.
        catd['vz'][i] = vz*100.
        catd['ID'][i] = IDs

    for i in range(ns):
        mass, x, y, z, vx, vy, vz, metals, tform, IDs = struct.unpack("<fffffffffQ", ffile.read(44))
        cats['mass'][i] = mass*2.324876e9
        cats['x'][i] = x
        cats['y'][i] = y
        cats['z'][i] = z
        cats['vx'][i] = vx*100.
        cats['vy'][i] = vy*100.
        cats['vz'][i] = vz*100.
        cats['metals'][i] = metals
        cats['tform'][i] = tform
        cats['ID'][i] = IDs
    return(catd,cats)