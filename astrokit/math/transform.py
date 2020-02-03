import numpy as np

def rot_2d(angle, pos_ax1, pos_ax2, ref_ax1 = 0.0, ref_ax2 = 0.0):

    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, s), (-s, c)))

    rot_ax1 = (R[0,0]* (pos_ax1-ref_ax1)+R[0,1]*(pos_ax2-ref_ax2))+ref_ax1
    rot_ax2 = (R[1,0]* (pos_ax1-ref_ax1)+R[1,1]*(pos_ax2-ref_ax2))+ref_ax2

    return rot_ax1, rot_ax2

def rectangle_edges(x0, y0, width, height):

    x=np.zeros([4])
    y=np.zeros([4])

    # left lower corner
    x[0]=x0+width/2.
    y[0]=y0-height/2.

    # left upper corner
    x[1]=x0+width/2.
    y[1]=y0+height/2.

    # left upper corner
    x[2]=x0-width/2.
    y[2]=y0+height/2.

    # right lower corner
    x[3]=x0-width/2.
    y[3]=y0-height/2.

    return (x,y)
