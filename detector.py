from nuRadioEmission import *
#rcParams['font.size']=14
nuRFEm = nuRadioEmission('moon')
from scipy.interpolate import interp1d
import numpy as np

def galactic_temperature(f_MHz):
    # Dulk 2001
    speed_of_light = 299792.458 # km / second
    nu = f_MHz # Hz
    tau = 5.0 * pow(nu, -2.1)
    Ieg = 1.0610e-20
    Ig  = 2.48e-20
    # Iv in  W/m^2/Hz/sr
    Iv = Ig * pow(nu, -0.52) * (1-np.exp(-tau))/tau + Ieg * pow(nu, -0.80) * np.exp(-tau)

    kB = 1.38064852e-23 # Watts / Hz / K
    c = speed_of_light * 1e3 # m/s
    temp = Iv * c**2 / (2*(nu*1e6)**2)/kB
    return Iv, temp # W/m^2/Hz/sr, K

class Detector:
    '''
    Frequencies are in MHz
    '''
    def __init__(self, geom, freq_lo=30., freq_hi=300., 
                 h_ant=1.0, a_ant=0.5e-2, 
                 T_amp = 500., ZL = 50.,
                 FE_type = 0, # 0 = transformer, 1 = resistive FET. 
                 impedance_ratio = 1., # for transformers these are typically square integer ratios
                 amp_noise_voltage = 1.2e-9, # V/sqrt(Hz) for FETs
                 N_ant_array = 1., # Number of antennas per polarization. Assume linear dipole array.
                 Nfrq=100, th_inc_cut_deg = 38.):
        '''
        Constants
        '''
        self.kB      = 1.38064852e-23 # Watts / Hz / K
        self.c_m_MHz = 299.792458 # speed of light in meter-MHz
        self.Z0      = 120.*np.pi # Ohm
        '''
        This cut needs to be in refracted angle
        '''
        self.th_inc_cut_deg = th_inc_cut_deg

        '''
        Detector Variables
        '''
        self.freq_lo = freq_lo
        self.freq_hi = freq_hi
        self.h_ant   = h_ant 
        self.T_amp   = T_amp
        self.a_ant   = a_ant 
        self.ZL      = ZL
        self.Nfrq    = Nfrq
        self.impedance_ratio = impedance_ratio
        self.N_ant_array = N_ant_array

        '''
        Initialize frequency array
        '''
        self.freq      = np.linspace(self.freq_lo, self.freq_hi, Nfrq, endpoint=True) # in MHz
        self.Nfrqs     = len(self.freq)
        self.df        = self.freq[1] - self.freq[0]
        self.lam_array = self.c_m_MHz / self.freq # in meters

        '''
        Initialize data arrays
        '''
        self.RE       = np.zeros((self.Nfrqs, geom.num_events))
        self.att_len  = np.zeros((self.Nfrqs, geom.num_events))
        self.att      = np.zeros((self.Nfrqs, geom.num_events))
        self.E_reg    = np.zeros((3, self.Nfrqs, geom.num_events))
        self.E_det    = np.zeros((3, self.Nfrqs, geom.num_events))
        self.V_det    = np.zeros((3, self.Nfrqs, geom.num_events))
        self.V_pk     = np.zeros((3, geom.num_events))
        self.V_gal_sq = np.zeros(self.Nfrqs)
        
        
        '''
        Initialize auxiliary inputs
        '''
        self.E_eV = 10**geom.proton_log10_energy
        self.energy_MeV = self.E_eV*1.e-6
        
        B_gal, self.T_gal = galactic_temperature(self.freq)
        #print(T_gal.shape)
        
        '''
        Amplifier noise contribution in V^2/MHz
        '''
        self.V_amp_sq = (self.kB*1.e6) * self.T_amp * np.real(self.ZL) * self.impedance_ratio # Noise voltage referenced to 50 Ohms
        if FE_type == 1:
            self.V_amp_sq = amp_noise_voltage**2  * 1.e6 # The factor of 1.e6 to is convert from V/sqrt(Hz) to V/sqrt(MHz) and squared.

        self.get_loss_tan(geom)
        
        
        #self.R_ant     = 20. * pi**2 * (self.h_ant/self.lam_array)**2
        #self.X_ant     = -120. * self.lam_array / np.pi / self.h_ant * (np.log(self.h_ant/2./self.a_ant)-1.)
        f_ant    = []
        R_ant    = []
        X_ant    = []
        maxD_ant = []
        '''Read Antenna Data (1m dipole, 1mm diameter rod)'''
        h_ant_string = '%1.0fm'%(h_ant)
        if h_ant%1.0 !=0.:         
            h_ant_string = '%1.1fm'%(h_ant)
            h_ant_string = h_ant_string.replace('.','p')

        fnm = 'dipole_data_%s.txt'%h_ant_string
        print('dipole file name:', fnm)
        #for line in file(fnm):
        f_dipole = open(fnm, "r")
        for line in f_dipole:
            f_ant.append(float(line.split()[0]))
            R_ant.append(float(line.split()[1]))
            X_ant.append(float(line.split()[2]))
            maxD_ant.append(float(line.split()[3]))
        f_dipole.close()
        
        iRa = interp1d(f_ant, R_ant)    
        iXa = interp1d(f_ant, X_ant)    
        imD = interp1d(f_ant, maxD_ant)
        
        self.R_ant    = iRa(self.freq)
        self.X_ant    = iXa(self.freq)
        self.maxD_ant = imD(self.freq)

        #self.V_div_fac = np.abs(self.ZL)/np.abs(self.ZL + self.R_ant + 1j*self.X_ant)
        '''NOTE THIS HAS TO GET SORTED OUT'''
        '''
        The electrically short antenna architecture depends on the amplifier noise and the voltage divider term. 
        '''
        self.V_div_fac = np.abs(self.ZL)**2/np.abs(self.ZL + self.R_ant + 1j*self.X_ant)**2
        if FE_type == 0:
            if np.abs(self.ZL-50.)>0.1 : 
                print('!!!! WARNING !!!!')
                print('Using a transformer with a non-50 Ohm amplifier is unusual')
                print('You have been warned.')
                print('!!!! WARNING !!!!')
            print('Calculating transformed impedance')
            ZL_tran = self.ZL * self.impedance_ratio
            self.V_div_fac = np.abs(ZL_tran)**2/np.abs(ZL_tran + self.R_ant + 1j*self.X_ant)**2
            
        self.h_eff     = self.N_ant_array * np.sqrt(4.* self.V_div_fac * self.R_ant/self.Z0 * self.lam_array**2/4./np.pi)
        
        '''
        Galactic noise spectrum in V^2/MHz
        '''
        self.V_gal_sq_spec = self.V_div_fac * 4. * self.R_ant * (self.kB*1.e6) * self.T_gal #*df
        self.V_rms         = np.sqrt(self.N_ant_array * np.sum(self.V_gal_sq_spec + self.V_amp_sq) * self.df)

        '''
        Effective height unit vectors
        '''
        r_hat = -geom.u_orb # direction of the radiation
        
        x_hat = np.zeros(r_hat.shape)
        y_hat = np.zeros(r_hat.shape)
        z_hat = np.zeros(r_hat.shape)
        
        # rotate the x-y plane randomly
        self.ant_az = np.random.uniform(0., 2.*np.pi, geom.num_events)

        x_hat[0,:] = +np.cos(self.ant_az)   
        x_hat[1,:] = +np.sin(self.ant_az)  
        y_hat[0,:] = -np.sin(self.ant_az) 
        y_hat[1,:] = +np.cos(self.ant_az) 
        z_hat[2,:] = 1.
        
        
        h_hat_x = x_hat - np.einsum('ij,ij->j',x_hat, r_hat)*r_hat
        h_hat_y = y_hat - np.einsum('ij,ij->j',y_hat, r_hat)*r_hat
        h_hat_z = z_hat - np.einsum('ij,ij->j',z_hat, r_hat)*r_hat
        
        h_hat_x /= np.einsum('ij,ij->j',h_hat_x, h_hat_x)
        h_hat_y /= np.einsum('ij,ij->j',h_hat_y, h_hat_y)
        h_hat_z /= np.einsum('ij,ij->j',h_hat_z, h_hat_z)
        
        # cosine along each direction
        cos_th_x = np.einsum('ij,ij->j', x_hat, r_hat)
        cos_th_y = np.einsum('ij,ij->j', y_hat, r_hat)
        cos_th_z = np.einsum('ij,ij->j', z_hat, r_hat)
        '''
        Frequency loop
        '''
        for k in range(self.Nfrqs):
            self.RE[k,:]      = nuRFEm.AMVZ(1., self.freq[k], geom.view_angle_rad, self.energy_MeV)
            self.att_len[k,:] = (self.lam_array[k]/(np.pi*geom.n_reg*self.loss_tan))
            self.att[k,:]     = np.exp(-geom.dist_reg*1.e3/self.att_len[k,:])
            
            self.E_reg[0,k,:] = self.RE[k,:]/(geom.dist_reg*1.e3)*self.att[k,:]*(geom.pol_vec_reg[0])
            self.E_reg[1,k,:] = self.RE[k,:]/(geom.dist_reg*1.e3)*self.att[k,:]*(geom.pol_vec_reg[1])
            self.E_reg[2,k,:] = self.RE[k,:]/(geom.dist_reg*1.e3)*self.att[k,:]*(geom.pol_vec_reg[2])

            self.E_det[0,k,:] = self.RE[k,:]*self.att[k,:] * (geom.e_perp * geom.u_pol_vec_perp_ref[0,:] + geom.e_para * geom.u_pol_vec_para_ref[0,:])
            self.E_det[1,k,:] = self.RE[k,:]*self.att[k,:] * (geom.e_perp * geom.u_pol_vec_perp_ref[1,:] + geom.e_para * geom.u_pol_vec_para_ref[1,:])
            self.E_det[2,k,:] = self.RE[k,:]*self.att[k,:] * (geom.e_perp * geom.u_pol_vec_perp_ref[2,:] + geom.e_para * geom.u_pol_vec_para_ref[2,:])

            '''
            First term is the E-field.
            Second term is the "sin theta" part of the radiation pattern.
            '''
            ''' TO DO: The x-y antennas need to sample a random azimuth'''
            # h = h_eff * D(\theta)
            # V = E \dot h_hat
            
            # Directivity (short dipole, update to general dipole formula)
            '''Short Dipole'''
            #Dx = 1.5*(1.-cos_th_x**2)
            #Dy = 1.5*(1.-cos_th_y**2)
            #Dz = 1.5*(1.-cos_th_z**2)
            
            ''' General Dipole, normalized using NEC2 outputs '''
            kL = (2.*np.pi/self.lam_array[k])*self.h_ant
            Dx = (np.cos(0.5*kL*cos_th_x)-np.cos(0.5*kL))**2/(1.-cos_th_x**2)
            Dy = (np.cos(0.5*kL*cos_th_y)-np.cos(0.5*kL))**2/(1.-cos_th_y**2)
            Dz = (np.cos(0.5*kL*cos_th_z)-np.cos(0.5*kL))**2/(1.-cos_th_z**2)
            
            norm = 10**(self.maxD_ant[k]/10.)/(1.-np.cos(0.5*kL))**2
            Dx *= norm
            Dy *= norm
            Dz *= norm
            
            #plot(cos_th_z, Dz, '.')
            
            V_x = np.einsum('ij,ij->j',self.E_det[:,k,:], h_hat_x) * self.h_eff[k] * np.sqrt(Dx)
            V_y = np.einsum('ij,ij->j',self.E_det[:,k,:], h_hat_y) * self.h_eff[k] * np.sqrt(Dy)
            V_z = np.einsum('ij,ij->j',self.E_det[:,k,:], h_hat_z) * self.h_eff[k] * np.sqrt(Dz)
            
            #self.V_det[0,k,:] = (np.cos(self.ant_az)*self.E_det[0,k,:] - np.sin(self.ant_az)*self.E_det[1,k,:]) * self.h_eff[k]
            #self.V_det[1,k,:] = (np.sin(self.ant_az)*self.E_det[0,k,:] + np.cos(self.ant_az)*self.E_det[1,k,:]) * self.h_eff[k]
            #self.V_det[2,k,:] = self.E_det[2,k,:] * self.h_eff[k]
            
            #self.V_det[0,k,:] = np.cos(self.ant_az) * V_x - np.sin(self.ant_az) * V_y
            #self.V_det[1,k,:] = np.sin(self.ant_az) * V_x + np.cos(self.ant_az) * V_y
            #self.V_det[2,k,:] = V_z

            self.V_det[0,k,:] = V_x 
            self.V_det[1,k,:] = V_y
            self.V_det[2,k,:] = V_z

            self.V_pk[0,:] += self.V_det[0,k,:]*self.df
            self.V_pk[1,:] += self.V_det[1,k,:]*self.df
            self.V_pk[2,:] += self.V_det[2,k,:]*self.df
            '''
            E_pk *= att**(1./lam_array[k])
            E_perp = E_pk * geom.e_perp * geom.u_pol_vec_perp_ref 
            E_para = E_pk * geom.e_para * geom.u_pol_vec_para_ref

            #V_pk[0,:] += np.sqrt(fac[k] * 4.*R_ant[k]/Z0 * lam_array[k]**2/4./pi) * E_pk * geom.e_perp * df
            #V_pk[1,:] += np.sqrt(fac[k] * 4.*R_ant[k]/Z0 * lam_array[k]**2/4./pi) * E_pk * geom.e_para * df
            V_pk[0,:] += np.sqrt(fac[k] * 4.*R_ant[k]/Z0 * lam_array[k]**2/4./pi) * (E_perp[0,:] + E_para[0,:]) 
            V_pk[1,:] += np.sqrt(fac[k] * 4.*R_ant[k]/Z0 * lam_array[k]**2/4./pi) * (E_perp[1,:] + E_para[1,:]) 
            V_pk[2,:] += np.sqrt(fac[k] * 4.*R_ant[k]/Z0 * lam_array[k]**2/4./pi) * (E_perp[2,:] + E_para[2,:]) 
            '''

        trig1 = geom.th_inc_rad*180./np.pi<self.th_inc_cut_deg  
        trig2 = np.logical_or(self.V_pk[0]>5.*self.V_rms, self.V_pk[1]>5.*self.V_rms)
        trig2 = np.logical_or(trig2, self.V_pk[2]>5.*self.V_rms)

        self.trig = np.logical_and(trig1, trig2)
    '''
    Function to initizalize the loss tangents for each event.
    '''
    def get_loss_tan(self, geom):
        '''
        Use a simple distribution of the Oxide fractions in the regolith (Mare, farside, etc.)
        '''
        self.M = np.zeros(geom.num_events) # this is %TiO2 + %FeO
        self.M[0:int(0.6*geom.num_events)] = 5.
        u = np.arange(int(0.6*geom.num_events), geom.num_events)
        self.M[int(0.6*geom.num_events):geom.num_events] = 5.+30.*(u-np.min(u))/(np.max(u)-np.min(u))
        '''
        Randomize the events Mare and highlands position of the events.
        '''
        np.random.shuffle(self.M)
        '''
        Estimate the loss tangents using the parameterization in the Lunar Source Book
        Note that this is the regolith density expected at a depth of a couple of meters. 
        It does vary, which is something that will have to be treated in a more sophisticated Monte Carlo.
        '''
        self.loss_tan = np.power(10., 0.038*self.M + 0.312*geom.density_reg - 3.26)
        #L_att_div_lam = (np.pi*(self.n_reg)*loss_tan)**-1.
        #att = np.exp(-geom.dist_reg*1.e3/L_att_div_lam)
        
        
        