# -*- coding: utf-8 -*-
"""
Created on Thu May 23 18:41:00 2024

@author: adrie
"""
import numpy as np

class geometry:
    def __init__(self):
        pass
    
    def draw_from_parametric(self):
        pass
    
    def draw_from_function(self,f):
        pass
    
    def get_rotation(self,point,center,theta):
        """
        compute the rotation in the plane of a tensor of index (matrix of x_index,matrix of y_index)
        """
        S=point.shape
        rotation_matrix=np.matrix([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        x_p=center[0]
        y_p=center[1]
        vec_point=np.array([point[0]-x_p,point[1]-y_p]).reshape(2,-1)
        #print(vec_point.min())
        A=np.array(rotation_matrix.dot(vec_point)).reshape(S)#[0]
        return A[0],A[1]
    
    
class circle(geometry):
    """
    The purpose of this class is to efficiently draw an elipse inside a numpy array
    Futhermore, elipses can be seen as the projection of a disk in a 3D space on a selected 2D plane
    here, we draw the elipse according to the tilt of the disk
    """
    def __init__(self,size_img,center=(0,0),r=100,angle_z=np.pi/4,angle_plan=np.pi/4,central_intensity=125):
        super().__init__()
        
        self.M=size_img # image is squared
        self.center=center #center of the circle
        self.r=r #radius of the circle
        self.boundaries=self.get_boundaries() #squared boundaries that contain the circle 
        self.angle_z=angle_z#tilt parameter for depth
        self.angle_plan=angle_plan#tilt parameter for rotation of the elipse along the projective plane
        self.central_intensity=central_intensity#luminance value for the pixel in the center of the
    def draw(self,img):
        return img
    
    def get_boundaries(self):
        """
        compute boundaries to limit number of computation
        we take care of edges as well as limit the image to draw the elipse inside the
        square that delimit the disk with 0 tilt for simplification
        """
        x_inf=int(np.round(max(self.center[0]-self.r,0)))
        x_sup=int(np.round(min(self.center[0]+self.r,self.M)))
        y_inf=int(np.round(max(self.center[1]-self.r,0)))
        y_sup=int(np.round(min(self.center[1]+self.r,self.M)))
        return x_inf,x_sup,y_inf,y_sup

    
    def draw(self,img):
        
        #first, compute the values of the elipsis radius according to tilt in depth
        new_x_r=np.abs(self.r*np.sin(-self.angle_z))
        new_y_r=self.r
        
        #get boundaries to extract subpart of the image relevant for computation
        x_inf,x_sup,y_inf,y_sup=self.boundaries
        
        #create 2 matrix, one for the x coordinates, 
        #the second for the y coordinates of each pixels inside the image
        #That way we can easily compute distance to a point in a parallelized manner
        #Step 1 we juste compute the arrays
        y_matrix=np.arange(y_inf,min(y_sup+1,img.shape[1]))[:,None]
        x_matrix=np.arange(x_inf,min(x_sup+1,img.shape[0]))[:,None]
        
        #condition to prevent cases where circle has a size lower than 1 pixel 
        if y_matrix.shape[0]>0 and x_matrix.shape[0]>0:
            #Step 2, we compute the matrix by replicating the array
            y_matrix=np.concatenate([y_matrix for i in range(len(x_matrix))],axis=-1)
            x_matrix=np.concatenate([x_matrix for i in range(len(y_matrix))],axis=-1).T
            
            #we then aggregate to get a tensor 
            F=np.concatenate([x_matrix[None,:,:],y_matrix[None,:,:]],axis=0)
            
            #we compute the rotation in the plane to get the new indexes
            x_proj,y_proj=self.get_rotation(F,self.center,self.angle_plan)
            
            x_matrix=x_proj
            y_matrix=y_proj
            
            #compute distance from center matrix, according to the ellipse equation
            d_matrix=(x_matrix/new_x_r)**2+(y_matrix/new_y_r)**2
            
            #fill the disk-elipse with 1
            d_matrix=d_matrix<=1
            
            #depth calculation
            z=(x_proj)/np.tan(self.angle_z)
            
            #function for the color to depend on depth where z is depth
            color=(self.central_intensity+2*z/self.M*self.central_intensity)*d_matrix
            
            #retrieve index of the filled disk
            #mask=img[y_inf:min(y_sup+1,img.shape[1]),x_inf:min(x_sup+1,img.shape[0])]>0
            
            # I don't remember, seems to be to prevent negative values which could occur in the coloring
            # also allow to draw multiple cirle on the same image without adding.
            patch1=color
            
            patch2=img[y_inf:min(y_sup+1,img.shape[1]),x_inf:min(x_sup+1,img.shape[0])]
            
            patch=np.concatenate([patch1[None,:,:],patch2[None,:,:]])
            
            patch=patch.max(axis=0)
            
            img[y_inf:min(y_sup+1,img.shape[1]),x_inf:min(x_sup+1,img.shape[0])]=patch
            
        return img
        
        
def make_crown(j,n_ellipses=5,dephasage=0,M=500,r=10,R=10,S=200):
    """
    this function draw multiple elipses along a circle. 
    Each of those is a central rotated copy of the previous one
    
    j is a parameter used in the animation to render dynamic growing and rotation
    R is the distance of the center of the ellipse to the center of the crown
    r is the nominal radius of the ellipse
    S is a scaling factor for the rotation, which depend of the number of frame
    """
    img=np.zeros((M,M))
    center=(int(M/2),(int(M/2)))
    #R=50#2+j/5
    #r=20#2+j/5
    LC=[]
    dephasage=np.pi*2#/360*j
    #compute parameters and instanciates classes
    for i in range(n_ellipses):
        angle=i*np.pi/n_ellipses*2+dephasage
        angle_plan=-i*np.pi/n_ellipses*2+dephasage#+np.pi/2/(360-j+1)
        angle_z=np.pi/2+j*np.pi*2/S*2
        center=(np.cos(angle)*R+int(M/2),np.sin(angle)*R+int(M/2))
        LC+=[circle(M,r=r,angle_plan=angle_plan,center=center,angle_z=angle_z)]
    #draw the ellipses by replacing    
    for C in LC:
        img=C.draw(img)
    return img