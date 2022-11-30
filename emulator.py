import numpy as np
from scipy.ndimage.filters import gaussian_filter
import MDAnalysis as mda
import matplotlib.pyplot as plt
import MDAnalysis.analysis.distances as distances
from PIL import Image


def cone_clash(pt,positions,r1,cone_angle):
    # radius of the tip (Angstrom)
    # r1=20
    
    # cone oriented along Z
    
    # where is the tip at the moment? (center of the sphere)
    # This is now :"pt"
    # opening angle of the tip (degrees)
    alpha=np.radians(cone_angle)
    tanalpha=np.tan(alpha)
    
    

    # if all point are below tip - ignore
    if np.all(positions[:,2]<pt[2]):
        return False
    else:
        rx=r1+(positions[:,2]-pt[2])*tanalpha
        
        # Compare squares to save time
        rx2=rx*rx
        dist=np.sqrt((positions[:,0]-pt[0])**2+(positions[:,1]-pt[1])**2)
        dist2=(positions[:,0]-pt[0])**2+(positions[:,1]-pt[1])**2

        if np.any(dist2<rx2):
            # if all clashing points are below the tip - does not matter
            if np.all(positions[np.where(dist<rx)][:,2]<pt[2]):
                return False
            else:
                # Real clash
                return True
        else:
            return False

def get_AFM3(uref2,frame,rotangle,rotaxis,rotpoint,stiffresidues=[1,1177],
             reso_x=5.0,reso_y=5.0,reso_z=5.0,probe_radius=20,noise_mn=0,
             noise_std=1.0,blur_x=1,blur_y=1,maxh=None,rawtiff=None,
             conical=True,cone_angle=0.0,field_px_x=200, field_px_y=200):
    """
    Calculate hsAFM images given a molecular system. This has been built to image SARS-CoV-2 spike protein 
    positioned on mica surface, therefore many options pertain to that. Nevertheless, this can be wasily adopted
    to any protein of interest or trajectory thereof.
    
    Options:
    uref2 - valid MDAnalysis Universe object containint preoriented protein
    frame - in case uref2 contains a molecular trajectory - frame on which the code will work
    rotangle, rotaxis, rotpoint - some basic utility to reorient the protein. Prefered option is to have protein preoriented beforehand
    stiffresidues - specific to the SARS-CoV-2 spike protein. Defines which residues are "stiff" and therefore visible to AFM, anything outside this range will be ignored
    reso_x, reso_y, reso_z - measurement resolution in x,y,z dimension # resolution of the AFM piezo (Angstrom)
    probe_radius - radius of the AFM tip curvature radius (Anstrom)
    conical - whether we consider the rest of the AFM tip or not. Bool.
    cone_angle - the opening angle of the cone emulating the AFM tip.
    noise_mn, noise_std - gaussian noise parameters emulating noise in z. Angstrom
    blur_x, blur_y - sigma of a gaussian blur applied to an image to emulate noise in xy
    maxh - mica surface height in z. If none - will be calculated from the mimimum of the Z coordinate of the protein
    rawtiff - name of the tiff file to be written. If none - skip.
    field_px_x, field_px_y - define field_of_view, the x-y span in pixels
    
    """
    
    field_of_view = [field_px_x, field_px_y] #


    # set the frame
    _=uref2.trajectory[frame]
    
    # Prepare the system - here we only look at CA atoms.
    selcalpha2=uref2.select_atoms('name CA and resid {}:{}'.format(stiffresidues[0],stiffresidues[1]))
    
    # apply transformations
    uref2.atoms.rotateby(rotangle,rotaxis,rotpoint)
    
    # Get span
    mins = np.min(uref2.atoms.positions,axis=0)
    maxs = np.max(uref2.atoms.positions,axis=0)

    # Calculate how much of an edge to add to get proper XY span. At the moment the 
    # resolution defines pixel size. px==reso
    newedge_x = (field_of_view[0]-np.ceil((maxs[0]-mins[0])/reso_x))*reso_x*0.5
    newedge_y = (field_of_view[1]-np.ceil((maxs[1]-mins[1])/reso_x))*reso_y*0.5
    
    
    # Create a grid over which we iterate.
    X=np.arange(mins[0]-newedge_x,maxs[0]+newedge_x,step=reso_x)
    Y=np.arange(mins[1]-newedge_y,maxs[1]+newedge_y,step=reso_y)
    Z=np.arange(mins[2],maxs[2]+1,step=reso_z)[::-1] # Invert Z because we want to go from the top to small values (spike is already upside down)    
    
    # Extent of the image in XY: position of X and Y arrays shifted to 0,0.
    # it should cover 0 to field_of_viev*reso
    extent=0.1*np.array([X[0]-np.min(X),X[-1]-np.min(X)+reso_x,Y[0]-np.min(Y),Y[-1]-np.min(Y)+reso_y])
 
    # Define an empty map
    height_map=np.zeros((X.shape[0],Y.shape[0]))

    # Assuming spike is upside down, we say mica starts where RBD ends. if maxh is defined - it is taken as mica
    if maxh is None:
        mica_surface=Z[-1]
    else:
        mica_surface=maxh.center_of_mass()[2]
        
        
    # Iterate over the grid
    for ix in range(len(X)):
        x=X[ix]

        for iy in range(len(Y)):
            y=Y[iy]
            for iz in range(len(Z)):
                z=Z[iz]

                # Skip all points which are far away from protein (helps to speed up if field of view is large)
                max_spread = 150 # this is how we predict a mol will be maximally spread by the afm tip convolution. If tip is very different - has to be adapted.
        
                if x<(mins[0]-max_spread) or x>(maxs[0]+max_spread) or \
                   y<(mins[1]-max_spread) or y>(maxs[1]+max_spread):
                        height_map[ix,iy]=mica_surface
                        # Far away - no detection
                        break
                
                
                pt=np.array([x,y,z])
                
                # Core of the clash detection here:
                
                
                # distance afm tip to protein
                d=distances.distance_array(pt,selcalpha2.positions)
                if np.min(d)<=probe_radius:
                    height_map[ix,iy]=z
                    break
                    
                # The tip is not clashing, but maybe the cone?
                elif conical:
                    if cone_clash(pt,selcalpha2.positions,probe_radius,cone_angle=cone_angle):
                        height_map[ix,iy]=z
                        break
                    
                # If we did not break so far, means we reached mica surface
                height_map[ix,iy]=mica_surface
    
    height_map=height_map-mica_surface
    
    # Add gaussian noise to emulate real AFM recording
    noise=np.random.normal(noise_mn,noise_std,height_map.shape)

    # Optional blur in XY with sigma equal to 1 pixel (this is our resolution)
    heigt_blurred = gaussian_filter(height_map, sigma=(blur_x,blur_y))
    hm_array=(heigt_blurred+noise)/10.
    
    # prep a Figure
    fig=plt.figure(figsize=(14,14))
    ax=fig.add_subplot(111)
    

    
    
    hm=ax.imshow(hm_array.T,cmap='copper',extent=extent,aspect=len(X)/float(len(Y)),origin='lower')
    ax.set_xlabel('x, nm')
    ax.set_ylabel('y, nm')
    
    cb=fig.colorbar(hm,ax=ax,use_gridspec=False,shrink=0.7)
    cb.set_label('height, nm')
    plt.show()
    
    # Write a raw tiff for further analysis
    if rawtiff is not None:
        
        maxH=100 # Theoretical max value in nm
        minH=np.min(hm_array)
        shifted_hm_array=hm_array-minH
        hhh=shifted_hm_array/maxH*255
        ahhh=hhh.astype(np.uint8)
        im = Image.fromarray(ahhh)
        
        # Make up some name that holds the info about conditions (very beautiful)
        imname=rawtiff.replace(".tif","")+"_min_{}_max_{}_resox_{}_resoy_{}.".format(minH,maxH,reso_x,reso_y)+\
                    "_resoz_{}_proberadius_{}_coneangle_{}".format(reso_z,probe_radius,cone_angle)+\
                    "_noisemn_{}_noisestd_{}_blurx_{}_blury_{}.tif".format(noise_mn,noise_std,blur_x,blur_y)
        im.save(imname)
    return hm_array
    
    
if __name__=="__main__":
   
   # example use case on a spike protein on mica. Sets parameters and calculates hsAFM image.
   outname='S_side_3open_1_1204_d15_nativeRBD.tif'
   updb=mda.Universe("pull_d_15_t500_tilt3_S_SIDE_nativeRBD.pdb",in_memory=True)
   frame=0
   rotangle=0
   rotaxis=[0,0,1]
   rotpoint=updb.atoms.center_of_mass()
   stiffresidues=[1,1204]
   probe_radius = 20
   reso_x  = 5
   reso_y  = 5
   reso_z  = 7
   noise_std = 3.0
   cone_angle = 30
   sigma = 1
   hm_cone=get_AFM3(updb,frame,rotangle,rotaxis,rotpoint,stiffresidues,noise_std=noise_std,reso_x=reso_x,
                           reso_y=reso_y,reso_z=reso_z,conical=True,cone_angle=cone_angle,blur_x=sigma,
                                        blur_y=sigma,rawtiff=outname,probe_radius=probe_radius)