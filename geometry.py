#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 17:06:16 2018

@author: romerowo
"""

import numpy as np
from scipy.interpolate import Akima1DInterpolator
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize


class Geometry:
    def __init__(self, altitude= 50., proton_log10_energy = 20., num_events=1000):
        self.altitude = altitude
        self.num_events = num_events
        self.proton_log10_energy = proton_log10_energy
        
        ''' Define the radius of the Moon '''
        self.Moon_radius_km = 1737.1
        
        '''
        The density and index of refraction of the regolith have a gradient 
        (see Lunar Source book)
        This will have to be treated in a more sophisticated Monte Carlo.
        Note: Alvarez-Muniz+ 2006 uses a density of 1.8 g/cm^3
        We are using a density of 1.4 g/cm^3 because this is relevant to shower
        development of UHECRs.
        We will have to update the radio emission model for the relevant 
        densities and index of refraction of regolith. 
        '''
        
        self.density_reg = 1.4 # g/cm^3
        
        '''
        The index of refraction relevant here is at the surface. 
        Alvarez-Muniz et al. used 1.73 but that is for deep regolith. The
        value relevant for refraction is 1.55 near the surface
        '''
        self.n_reg = 1.55 # index of refraction of the lunar regolith
        
        '''
        The geometric factor is obtained from integrating the acceptance 
        assuming every event is visible. This is the scaling factor for the
        efficiency.
        '''
        
        self.AOmega_0 = 2.*np.pi**2 * self.Moon_radius_km * self.altitude / (1.+self.altitude/self.Moon_radius_km) # km^2 sr

        ''' Calculate the horizon latitude angle '''
        
        self.cos_horz = (1.+self.altitude/self.Moon_radius_km)**(-1)
        
        
        '''
        # NEW CODE HERE
        1. Generate events: theta_M, theta_CR, phi_CR
        2. Propagate Particle
        3. Propagate radio path
        '''
        
        self.generate_events()
        
        #'''Get random interaction points'''
        #
        #self.cos_th_M = np.random.uniform(self.cos_horz, 1., self.num_events)
        #self.sin_th_M = np.sqrt(1.-self.cos_th_M**2)
        
        #''' The entry point of the cosmic ray in Cartesian coordinates '''
        #self.r_M = np.zeros((3, self.num_events))
        #self.r_M[0,:] = self.sin_th_M 
        #self.r_M[2,:] = self.cos_th_M 

        
        #'''
        #Get random directions.
        #Note, we sample the cosmic ray zenith angle in cos^2 (not cos) to 
        #take the dot product in the integral into account. This angle is 
        #defined in a coordinate system with the z-axis along the normal to 
        #the surface of the Moon.
        #'''
        #self.cos_th_CR = np.sqrt(np.random.uniform(0., 1., self.num_events))
        #self.phi_CR    = np.random.uniform(0., 2.*np.pi, self.num_events)

        #'''
        #This is the direction unit vector of the particle  in a coordinate 
        #system with the z-axis defined along the direction of the normal to 
        #the Moon. 
        #'''
        #k_x = np.sqrt(1.-self.cos_th_CR**2) * np.cos(self.phi_CR)
        #k_y = np.sqrt(1.-self.cos_th_CR**2) * np.sin(self.phi_CR)
        #k_z = self.cos_th_CR
        
        #'''
        #This is the direction unit vector in the coordinate system of the 
        #spacecraft (S/C is in the z-axis of the Moon)
        #'''
        #self.r_CR = np.zeros((3, self.num_events))
        #self.r_CR[0,:] = +k_x*self.cos_th_M + k_z*self.sin_th_M
        #self.r_CR[1,:] =  k_y
        #self.r_CR[2,:] = -k_x*self.sin_th_M + k_z*self.cos_th_M
        

        #'''
        #Calculate the zenith angle of the event w.r.t. the normal to the surface 
        #'''
        ##self.cos_theta_zen =  self.r_CR[0,:]*self.r_M[0,:] \
        ##                    + self.r_CR[1,:]*self.r_M[1,:] \
        ##                    + self.r_CR[2,:]*self.r_M[2,:] 
              
        '''
        > Particle interaction model
        This is a model of Xmax for given energies. 
        It was a by-eye estimate of a plot with various data points.
        Will want a citeable model with Xmax and sigma-Xmax.
        This simple model will have to do for now. 
        '''
        
        self.proton_log10_energy_list = np.array([14.,   15.,   16.,   17.,   18.,   19.,   20.,   21.01])
        self.proton_Xmax_list         = np.array([500., 580.,  640.,  690.,  760.,  800.,  845.,  880.]) # in g/cm^2
        self.get_proton_Xmax          = Akima1DInterpolator(self.proton_log10_energy_list, self.proton_Xmax_list)

        #self.shower_lam = self.get_proton_Xmax(self.proton_log10_energy) / self.density_reg * 1.e-5 # in km
        #self.shower_pos = self.Moon_radius_km * self.r_M - self.shower_lam * self.r_CR

        self.propagate_particle()

        self.propagate_radio()
        #for k in range(0,self.num_events):
        self.R_SC = np.array([0., 0., self.Moon_radius_km + self.altitude])
        e1 = self.R_SC/np.sqrt(np.dot(self.R_SC,self.R_SC))
        e2 = np.zeros((3,self.num_events))
        #proj = (self.R_SC[0]*self.shower_pos[0,:] + self.R_SC[1]*self.shower_pos[1,:] + self.R_SC[2]*self.shower_pos[2,:])/(np.dot(self.R_SC,self.R_SC))
        proj = np.einsum('i,ij->j', self.R_SC, self.shower_pos)/(np.dot(self.R_SC,self.R_SC))
        e2 = self.shower_pos - proj*np.outer(self.R_SC, np.ones(self.num_events))
        #e2 /= np.sqrt(e2[0,:]**2 + e2[1,:]**2 + e2[2,:]**2)
        e2 /=  np.sqrt(np.einsum('ij,ij->j',e2,e2))
        self.R_Tx = np.zeros((3,self.num_events))

        self.snell_residual = []
        '''
        for k in range(0,self.num_events):
            # we define the light travel time. The minimum path is correct.
            def ct(x):
                #1. Rotate shower position so that y-component is 0.
                phi = np.arctan(self.shower_pos[1,k]/self.shower_pos[0,k])
                r_1 = np.array(
                       [+np.cos(phi)*self.shower_pos[0,k]+np.sin(phi)*self.shower_pos[1,k],
                       -np.sin(phi)*self.shower_pos[0,k]+np.cos(phi)*self.shower_pos[1,k],
                       self.shower_pos[2,k]]
                       )
                r_3 = self.R_SC.copy()
                r_2 = [x, 0.,  np.sqrt(self.Moon_radius_km**2-x**2)]
                r_12 = np.sqrt(np.dot(r_2-r_1,  r_2-r_1))
                r_23 = np.sqrt(np.dot(r_2-r_3,  r_2-r_3))
                return  r_12 + self.n_reg*r_23
            res = minimize(ct, x0=[0.])
            self.R_Tx[:,k] = np.sqrt(self.Moon_radius_km **2-res.x**2) * e1 + res.x * e2[:,k]
        '''
        #'''
        for k in range(0,self.num_events):
            # we define the light travel time. The minimum path is correct.
            def ct(x):
                R_Tx = np.sqrt(self.Moon_radius_km**2 - x**2) * e1 + x * e2[:,k]
                r_TX_SC = np.sqrt(np.dot(self.R_SC-R_Tx,  self.R_SC-R_Tx))
                r_TX_sh = np.sqrt(np.dot(self.shower_pos[:,k]-R_Tx, self.shower_pos[:,k]-R_Tx))
                return  r_TX_SC + self.n_reg*r_TX_sh
            res = minimize_scalar(ct, tol=1.e-12)
            self.R_Tx[:,k] = np.sqrt(self.Moon_radius_km **2-res.x**2) * e1 + res.x * e2[:,k]
            
            R_Tx = self.R_Tx[:,k]
            c1 = np.cross(R_Tx, self.R_SC-R_Tx) 
            c2 = np.cross(R_Tx, self.shower_pos[:,k]-R_Tx)
            d1 = R_Tx - self.R_SC
            d2 = R_Tx - self.shower_pos[:,k]
            mc1 = np.sqrt(np.dot(c1, c1))
            mc2 = np.sqrt(np.dot(c2, c2))
            md1 = np.sqrt(np.dot(d1, d1))
            md2 = np.sqrt(np.dot(d2, d2))
            #val = mc1/md1 - self.n_reg * mc2/md2
            #print x, mc2, md1, mc1/md1 * md2/mc2, self.n_reg 
            val = mc1/md1 * md2/mc2 - self.n_reg 

            self.snell_residual.append(val)

            #print 'a: ', res.x
        #'''
        '''
        for k in range(0,self.num_events):
            def snell(x):
                R_Tx = np.sqrt(self.Moon_radius_km**2 - x**2) * e1 + x * e2[:,k]
                c1 = np.cross(R_Tx, self.R_SC-R_Tx) 
                c2 = np.cross(R_Tx, self.shower_pos[:,k]-R_Tx)
                d1 = R_Tx - self.R_SC
                d2 = R_Tx - self.shower_pos[:,k]
                mc1 = np.sqrt(np.dot(c1, c1))
                mc2 = np.sqrt(np.dot(c2, c2))
                md1 = np.sqrt(np.dot(d1, d1))
                md2 = np.sqrt(np.dot(d2, d2))
                #val = mc1/md1 - self.n_reg * mc2/md2
                #print x, mc2, md1, mc1/md1 * md2/mc2, self.n_reg 
                val = mc1/md1 * md2/mc2 - self.n_reg 
                return np.abs(val)
            
            x0 = self.shower_pos[0,k]-1.
            x1 = self.shower_pos[0,k]
            #res = minimize_scalar(snell, bounds=(x0, x1), method='bounded')
            #print 'res.x', res.x
            #res = minimize(snell, res.x, tol=1.e-6)
            #print 'self.shower_pos[0,k]-5.e-3', self.shower_pos[0,k]-5.e-3
            res = minimize(snell, [self.shower_pos[0,k]-5.e-3], method='TNC', bounds = [(x0,x1)], tol=1.e-6)
            # while res.fun>1.e-1:
            #     x_guess = np.random.uniform(x0,x1)
            #     res = minimize(snell, [x_guess], method='TNC', bounds = [(x0,x1)], tol=1.e-6)
                
            #print '%1.2e'%res.fun, res.success
            self.snell_residual.append(res.fun)
            print 'b:', res.x, res.success
            #self.R_Tx[:,k] = res.x*e1 + np.sqrt(self.Moon_radius_km **2-res.x**2)*e2[:,k]
            self.R_Tx[:,k] = np.sqrt(self.Moon_radius_km **2-res.x**2) * e1 + res.x * e2[:,k]

            #print 'b: res.x', res.x, res.success
        #dot = (self.r_CR[0,:]*(self.R_Tx - self.shower_pos)[0,:]) + (self.r_CR[1,:]*(self.R_Tx - self.shower_pos)[1,:]) + (self.r_CR[2,:]*(self.R_Tx - self.shower_pos)[2,:])
        '''
        '''
        Cut events from the trigger if the snell residuals suck.
        This usually happens at nadir or by the horizon.
        '''
        self.snell_residual = np.array(self.snell_residual)
        '''
        This is the unit vector pointing from the shower to the refraction
        point on the surface. 
        '''
        self.u_reg = self.R_Tx - self.shower_pos
        self.dist_reg = np.sqrt(np.einsum('ij,ij->j', self.u_reg, self.u_reg)) 
        self.u_reg /= self.dist_reg
        
        '''a
        The cosine of the view angle is the dot product of the direction of
        propagation (-r_CR) and the direction from the shower to the 
        refraction point.
        '''
        dot = np.einsum('ij,ij->j',-self.r_CR, self.u_reg) 
        self.view_angle_rad = np.arccos(dot)

        '''
        The polarization vector is the projected direction of the shower 
        (times -1 since the charge excess is negative) given by r_CR and the
        direction of the radio propagation from the shower to the refraction  
        point on the surface
        '''        
        self.pol_vec_reg= -self.r_CR - dot * self.u_reg
        self.pol_vec_reg /= np.sqrt(np.einsum('ij,ij->j',self.pol_vec_reg, self.pol_vec_reg))
        
        '''
        These values are needed to estimate the transition coefficient. 
        The first thing needed are the polarization vector component 
        (in regolith) parallel and orthogonal to the normal to the surface 
        '''
        self.u_pol_vec_perp = self.R_Tx/np.sqrt(np.einsum('ij,ij->j', self.R_Tx, self.R_Tx))
        
        self.u_pol_vec_para = np.zeros(self.u_pol_vec_perp.shape)
        self.u_pol_vec_para[0,:] = self.u_reg[1,:]*self.u_pol_vec_perp[2,:] - self.u_reg[2,:]*self.u_pol_vec_perp[1,:] 
        self.u_pol_vec_para[1,:] = self.u_reg[2,:]*self.u_pol_vec_perp[0,:] - self.u_reg[0,:]*self.u_pol_vec_perp[2,:]
        self.u_pol_vec_para[2,:] = self.u_reg[0,:]*self.u_pol_vec_perp[1,:] - self.u_reg[1,:]*self.u_pol_vec_perp[0,:]
        self.u_pol_vec_para /= np.sqrt(np.einsum('ij,ij->j', self.u_pol_vec_para, self.u_pol_vec_para))

        '''
        The polarization vector after refraction is given by the polarization
        vector prior to refraction projected in the direction from the 
        refraction point to the spacecraft.
        ref stands for refracted
        '''
        self.u_orb = np.outer(self.R_SC, np.ones(self.num_events)) - self.R_Tx
        self.u_orb /= np.sqrt(np.einsum('ij,ij->j', self.u_orb, self.u_orb))

        self.u_pol_vec_perp_ref = self.u_pol_vec_perp.copy()
        self.u_pol_vec_perp_ref -= np.einsum('ij,ij->j',self.u_pol_vec_perp, self.u_orb)*self.u_orb
        self.u_pol_vec_perp_ref /= np.sqrt(np.einsum('ij,ij->j',self.u_pol_vec_perp_ref, self.u_pol_vec_perp_ref))

        self.u_pol_vec_para_ref = self.u_pol_vec_para.copy()
        self.u_pol_vec_para_ref -= np.einsum('ij,ij->j',self.u_pol_vec_para, self.u_orb)*self.u_orb
        self.u_pol_vec_para_ref /= np.sqrt(np.einsum('ij,ij->j',self.u_pol_vec_para_ref, self.u_pol_vec_para_ref))

        
        
        '''
        The projection factors for the electric field magnitude in the 
        directions parallel and sperpendicular to the surface.
        '''
        #self.e_perp = np.einsum('ij,ij->j', self.u_reg, self.u_pol_vec_perp)
        #self.e_para = np.einsum('ij,ij->j', self.u_reg, self.u_pol_vec_para)
        self.e_perp = np.einsum('ij,ij->j', self.pol_vec_reg, self.u_pol_vec_perp)
        self.e_para = np.einsum('ij,ij->j', self.pol_vec_reg, self.u_pol_vec_para)
        
        '''
        The modified transmission coefficients (See Lehtinen+ 2003, FORTE 
        satelline constraints... Appendix 2).
        '''
        
        #cos_ref = np.einsum('ij,ij->j',self.u_pol_vec_perp, np.outer(self.R_SC, np.ones(self.num_events)) - self.R_Tx)
        #cos_ref /= np.sqrt(np.einsum('ij,ij->j',np.outer(self.R_SC, np.ones(self.num_events)) - self.R_Tx, np.outer(self.R_SC, np.ones(self.num_events)) - self.R_Tx))

        #cos_inc = np.einsum('ij,ij->j',self.u_pol_vec_perp, self.R_Tx - self.shower_pos )
        #cos_inc /= np.sqrt(np.einsum('ij,ij->j', self.R_Tx - self.shower_pos, self.R_Tx - self.shower_pos))

        self.cos_ref  =  np.einsum('ij,ij->j',self.R_Tx, np.outer(self.R_SC, np.ones(self.num_events))-self.R_Tx)
        self.cos_ref /= np.sqrt(np.einsum('ij,ij->j',self.R_Tx, self.R_Tx)*np.einsum('ij,ij->j',np.outer(self.R_SC, np.ones(self.num_events))-self.R_Tx, np.outer(self.R_SC, np.ones(self.num_events))-self.R_Tx))
        
        self.cos_inc  = np.einsum('ij,ij->j',-self.R_Tx, self.shower_pos -self.R_Tx)
        self.cos_inc /= np.sqrt(np.einsum('ij,ij->j',-self.R_Tx, -self.R_Tx) * np.einsum('ij,ij->j',self.shower_pos -self.R_Tx,self.shower_pos -self.R_Tx))
 

        self.th_inc_rad = np.arccos(self.cos_inc)        

        self.e_perp *= 2.*self.cos_ref/(self.n_reg * self.cos_inc + self.cos_ref)
        self.e_para *= 2.*self.cos_ref/(self.n_reg * self.cos_ref + self.cos_inc)
        
        '''
        Finally, these are divided by the distance from the transmission point 
        to the spacecraft. So all we need to do to get the electric field 
        vector at the antenna is to multiply it by these coefficients.
        '''
        self.R_surf_to_SC = np.sqrt(np.einsum('ij,ij->j',np.outer(self.R_SC, np.ones(self.num_events)) - self.R_Tx, np.outer(self.R_SC, np.ones(self.num_events)) - self.R_Tx))

        self.e_perp *= 1./(self.R_surf_to_SC*1.e3) # in units of 1/m
        self.e_para *= 1./(self.R_surf_to_SC*1.e3) # in units of 1/m
        
        self.pol_vec_ref = self.e_para * self.u_pol_vec_para_ref + self.e_perp * self.u_pol_vec_perp_ref
        self.pol_vec_ref /= np.sqrt(np.einsum('ij,ij->j', self.pol_vec_ref, self.pol_vec_ref))
        
        
    ###########################################################################
    ###########################################################################
    ###########################################################################

    def generate_events(self):
        '''
        > Get random interaction points
        Use rotational symmetry to only generate impact points in lunar latitude angle.
        Azimuth gets randomized and not necessary for the calculations here
        '''
        self.th_M = np.arccos(np.random.uniform(self.cos_horz, 1., self.num_events))      
        '''
        > Get random directions.
        Note, we sample the cosmic ray zenith angle in cos^2 (not cos) to 
        take the dot product in the integral into account. This angle is 
        defined in a coordinate system with the z-axis along the normal to 
        the surface of the Moon.
        '''
        self.th_CR     = np.arccos(np.sqrt(np.random.uniform(0., 1., self.num_events)))
        self.phi_CR    = np.random.uniform(0., 2.*np.pi, self.num_events)

    ###########################################################################
    ###########################################################################
    ###########################################################################

    def propagate_particle(self):
        ''' 
        > CR entry point geometry
        Calculate the sine and cosine of the lunar latitude angle of CR entry
        Then get the CR entry point in Cartesian coordinates
        '''
        self.cos_th_M = np.cos(self.th_M)
        self.sin_th_M = np.sqrt(1.-self.cos_th_M**2)
        
        self.r_M = np.zeros((3, self.num_events))
        self.r_M[0,:] = self.sin_th_M 
        self.r_M[2,:] = self.cos_th_M 
        
        '''
        > CR direction unit vector in local coordinate system
        This is the direction unit vector of the particle  in a coordinate 
        system with the z-axis defined along the direction of the normal to 
        the Moon. 
        '''
        self.cos_th_CR = np.cos(self.th_CR)
        k_x = np.sqrt(1.-self.cos_th_CR**2) * np.cos(self.phi_CR)
        k_y = np.sqrt(1.-self.cos_th_CR**2) * np.sin(self.phi_CR)
        k_z = self.cos_th_CR
        
        '''
        > CR direction unit vector in global coordinate system
        This is the direction unit vector in the coordinate system of the 
        spacecraft (S/C is in the z-axis of the Moon)
        '''
        self.r_CR = np.zeros((3, self.num_events))
        self.r_CR[0,:] = +k_x*self.cos_th_M + k_z*self.sin_th_M
        self.r_CR[1,:] =  k_y
        self.r_CR[2,:] = -k_x*self.sin_th_M + k_z*self.cos_th_M
        
        '''
        > Get proton Xmax coordinates
        The shower Xmax is propagated to get shower position coordinates (shower_pos)
        '''
        self.shower_lam = self.get_proton_Xmax(self.proton_log10_energy) / self.density_reg * 1.e-5 # in km
        self.shower_pos = self.Moon_radius_km * self.r_M - self.shower_lam * self.r_CR

    ###########################################################################
    ###########################################################################
    ###########################################################################

    def propagate_radio(self):
        return 0.